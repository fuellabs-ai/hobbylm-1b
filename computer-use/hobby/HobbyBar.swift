// Hobby bar: a Spotlight-style panel for Hobby. Press Shift-Command-H anywhere, type a request, press Enter.
//
//   swiftc -O hobby/HobbyBar.swift -o data/HobbyBar && data/HobbyBar
// Runs hobby/hobby_server.py (the warm model) as a child process and shows each decision as it happens.
// The panel never activates its own app, so the app you were in stays frontmost for the model to see.
// Esc hides the panel. HOBBY_BAR_HOTKEY = cmd-shift-h (default, ⇧⌘H) |
// cmd-h (replaces the system Hide shortcut while Hobby runs) | ctrl-option-space | option-space | cmd-shift-space.
// --selftest "request || request": opens the panel, submits each request in code (no keyboard events), saves a picture of the panel
// window only and the transcript to data/hobby-bar-selftest.{png,txt}, then quits.
import AppKit
import Carbon.HIToolbox

func rgb(_ hex: Int, _ alpha: CGFloat = 1) -> NSColor {
    NSColor(srgbRed: CGFloat(hex >> 16 & 0xFF) / 255, green: CGFloat(hex >> 8 & 0xFF) / 255, blue: CGFloat(hex & 0xFF) / 255, alpha: alpha)
}

// fuellabs. palette: Paper, Ink, Lime, Muted, Line (+ two tints derived from them for the card and chips)
let card = rgb(0xFFFFFF), ink = rgb(0x11110F), lime = rgb(0xB7F34A), muted = rgb(0x66665F), line = rgb(0xD8D4C9)
let hairline = rgb(0xECEAE4), limeSoft = rgb(0xEEF8DA), olive = rgb(0x4A6614), chipFill = rgb(0xF3F2EE)
let panelWidth: CGFloat = 440, indent: CGFloat = 0  // transcript text starts at the panel inset, under the traffic lights

func ui(_ size: CGFloat, _ weight: NSFont.Weight = .regular) -> NSFont { .systemFont(ofSize: size, weight: weight) }
func mono(_ size: CGFloat, _ weight: NSFont.Weight = .medium) -> NSFont { .monospacedSystemFont(ofSize: size, weight: weight) }

func label(_ text: String, _ font: NSFont, _ color: NSColor, wrap: Bool = true) -> NSTextField {
    let l = wrap ? NSTextField(wrappingLabelWithString: text) : NSTextField(labelWithString: text)
    l.font = font
    l.textColor = color
    l.isSelectable = true
    l.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
    return l
}

/// A rounded pill: action chips in the transcript and the model chips in the bottom bar.
final class Pill: NSView {
    let text: NSTextField
    init(_ s: String, font: NSFont, fill: NSColor, color: NSColor = ink, border: NSColor? = nil) {
        text = NSTextField(labelWithString: s)
        super.init(frame: .zero)
        wantsLayer = true
        layer?.backgroundColor = fill.cgColor
        layer?.cornerRadius = 7
        if let b = border { layer?.borderColor = b.cgColor; layer?.borderWidth = 1 }
        text.font = font
        text.textColor = color
        text.translatesAutoresizingMaskIntoConstraints = false
        addSubview(text)
        NSLayoutConstraint.activate([
            text.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 8),
            text.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -8),
            text.topAnchor.constraint(equalTo: topAnchor, constant: 3),
            text.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -3),
        ])
        setContentHuggingPriority(.required, for: .horizontal)
        setContentCompressionResistancePriority(.required, for: .horizontal)
    }
    required init?(coder: NSCoder) { nil }
}

/// A pill-shaped button (calendar confirmation).
final class PillButton: NSButton {
    var onClick: () -> Void = {}
    init(_ title: String, filled: Bool) {
        super.init(frame: .zero)
        isBordered = false
        wantsLayer = true
        layer?.cornerRadius = 7
        layer?.backgroundColor = (filled ? ink : chipFill).cgColor
        attributedTitle = NSAttributedString(string: title, attributes: [.font: ui(12, .medium), .foregroundColor: filled ? card : ink])
        target = self
        action = #selector(fire)
        translatesAutoresizingMaskIntoConstraints = false
        heightAnchor.constraint(equalToConstant: 24).isActive = true
        widthAnchor.constraint(greaterThanOrEqualToConstant: 72).isActive = true
    }
    required init?(coder: NSCoder) { nil }
    @objc func fire() { onClick() }
}

/// Terminal-style window controls: red hides, yellow collapses to the input line, green makes the transcript taller.
/// The symbols appear on hover, as in macOS.
final class TrafficLights: NSView {
    let buttons: [NSButton]
    let glyphs = ["×", "−", "+"]
    init(_ actions: [() -> Void]) {
        let colors = [rgb(0xFF5F57), rgb(0xFEBC2E), rgb(0x28C840)]
        buttons = zip(colors, actions).map { color, act in
            let b = ClosureButton(act)
            b.isBordered = false
            b.wantsLayer = true
            b.layer?.cornerRadius = 6
            b.layer?.backgroundColor = color.cgColor
            b.layer?.borderWidth = 0.5
            b.layer?.borderColor = NSColor.black.withAlphaComponent(0.12).cgColor
            b.title = ""
            b.translatesAutoresizingMaskIntoConstraints = false
            NSLayoutConstraint.activate([b.widthAnchor.constraint(equalToConstant: 12), b.heightAnchor.constraint(equalToConstant: 12)])
            return b
        }
        super.init(frame: .zero)
        let row = NSStackView(views: buttons)
        row.spacing = 8
        row.translatesAutoresizingMaskIntoConstraints = false
        addSubview(row)
        NSLayoutConstraint.activate([
            row.leadingAnchor.constraint(equalTo: leadingAnchor), row.trailingAnchor.constraint(equalTo: trailingAnchor),
            row.topAnchor.constraint(equalTo: topAnchor), row.bottomAnchor.constraint(equalTo: bottomAnchor)])
        addTrackingArea(NSTrackingArea(rect: .zero, options: [.mouseEnteredAndExited, .activeAlways, .inVisibleRect], owner: self))
    }
    required init?(coder: NSCoder) { nil }
    override func mouseEntered(with e: NSEvent) { setGlyphs(true) }
    override func mouseExited(with e: NSEvent) { setGlyphs(false) }
    func setGlyphs(_ on: Bool) {
        for (b, g) in zip(buttons, glyphs) {
            b.attributedTitle = NSAttributedString(string: on ? g : "", attributes: [
                .font: NSFont.systemFont(ofSize: 10, weight: .bold), .foregroundColor: NSColor.black.withAlphaComponent(0.55)])
        }
    }
}

final class ClosureButton: NSButton {
    let run: () -> Void
    init(_ run: @escaping () -> Void) {
        self.run = run
        super.init(frame: .zero)
        target = self
        action = #selector(fire)
    }
    required init?(coder: NSCoder) { nil }
    @objc func fire() { run() }
}

final class Flipped: NSView { override var isFlipped: Bool { true } }

final class BarPanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override func cancelOperation(_ sender: Any?) { orderOut(nil) }  // Esc
}

final class Hobby: NSObject, NSApplicationDelegate, NSTextFieldDelegate {
    enum Mode { case loading, ready, busy, asking, stopped }

    let root: URL
    let panel = BarPanel(contentRect: NSRect(x: 0, y: 0, width: panelWidth, height: 200),
                         styleMask: [.nonactivatingPanel, .borderless], backing: .buffered, defer: false)
    let field = NSTextField()
    let statusDot = NSView()
    let status = label("Loading", ui(11), muted, wrap: false)
    let transcript = NSStackView()
    let scroll = NSScrollView()
    let thinking = NSStackView()
    let thinkingText = label("Thinking", ui(13), muted, wrap: false)
    let send = NSButton()
    var scrollHeight: NSLayoutConstraint!
    var lastStep: NSStackView?
    var transcriptText = ""
    var dots = 0, dotTimer: Timer?
    var doneSeconds: Double?
    var resizing = false  // our own setFrame calls are not user drags
    var collapsed = false, tall = false
    var maxTranscript: CGFloat { tall ? 560 : 300 }
    let clear = NSButton()

    var statusItem: NSStatusItem?
    var worker: Process?
    var toWorker: FileHandle?
    var buffer = Data()
    var mode = Mode.loading
    var hotKey: EventHotKeyRef?
    var hotKeyStatus = "not registered"
    let selftest: String? = {
        let a = CommandLine.arguments
        guard let i = a.firstIndex(of: "--selftest"), i + 1 < a.count else { return nil }
        return a[i + 1]
    }()

    init(root: URL) { self.root = root }

    func applicationDidFinishLaunching(_ note: Notification) {
        buildPanel()
        buildMenu()
        registerHotKey()
        startWorker()
        if CommandLine.arguments.contains("--show") { show() }
    }

    // MARK: window

    func buildPanel() {
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.hidesOnDeactivate = false
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.isMovableByWindowBackground = true  // drag by the header or any empty area
        NotificationCenter.default.addObserver(forName: NSWindow.didMoveNotification, object: panel, queue: .main) { [weak self] _ in
            guard let self, self.panel.isVisible, !self.resizing else { return }
            UserDefaults.standard.set(NSStringFromPoint(NSPoint(x: self.panel.frame.minX, y: self.panel.frame.maxY)), forKey: "hobbyBarTopLeft")
        }

        let box = NSView()
        box.wantsLayer = true
        box.layer?.backgroundColor = card.cgColor
        box.layer?.cornerRadius = 21
        box.layer?.borderWidth = 1
        box.layer?.borderColor = hairline.cgColor
        box.layer?.masksToBounds = true
        panel.contentView = box

        // header: lime square + "Hobby" ........ ● Offline · ready
        let lights = TrafficLights([
            { [weak self] in self?.panel.orderOut(nil) },
            { [weak self] in self?.toggleCollapsed() },
            { [weak self] in self?.toggleTall() },
        ])
        let square = NSView()  // the lime full stop, as in the fuellabs. wordmark
        square.wantsLayer = true
        square.layer?.backgroundColor = lime.cgColor
        square.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([square.widthAnchor.constraint(equalToConstant: 4), square.heightAnchor.constraint(equalToConstant: 4)])
        statusDot.wantsLayer = true
        statusDot.layer?.cornerRadius = 3.5
        statusDot.layer?.backgroundColor = line.cgColor
        statusDot.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([statusDot.widthAnchor.constraint(equalToConstant: 7), statusDot.heightAnchor.constraint(equalToConstant: 7)])
        let spacer = NSView()
        spacer.setContentHuggingPriority(.init(1), for: .horizontal)
        let title = label("Hobby", ui(13, .medium), ink, wrap: false)
        title.isSelectable = false  // header labels are drag handles, not text
        status.isSelectable = false
        clear.isBordered = false
        clear.target = self
        clear.action = #selector(clearChat)
        setClearEnabled(false)
        let titleRow = NSStackView(views: [title, square])
        titleRow.spacing = 1
        titleRow.alignment = .lastBaseline
        let header = NSStackView(views: [lights, titleRow, spacer, statusDot, status, clear])
        header.spacing = 7
        header.setCustomSpacing(12, after: lights)
        header.setCustomSpacing(14, after: status)
        header.alignment = .centerY

        // transcript (scrolls once it is taller than maxTranscript)
        transcript.orientation = .vertical
        transcript.alignment = .leading
        transcript.spacing = 8
        transcript.edgeInsets = NSEdgeInsets(top: 0, left: indent, bottom: 0, right: 0)
        transcript.translatesAutoresizingMaskIntoConstraints = false
        let doc = Flipped()
        doc.translatesAutoresizingMaskIntoConstraints = false
        doc.addSubview(transcript)
        scroll.documentView = doc
        scroll.drawsBackground = false
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        scroll.borderType = .noBorder
        scroll.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            transcript.leadingAnchor.constraint(equalTo: doc.leadingAnchor),
            transcript.trailingAnchor.constraint(equalTo: doc.trailingAnchor),
            transcript.topAnchor.constraint(equalTo: doc.topAnchor),
            transcript.bottomAnchor.constraint(equalTo: doc.bottomAnchor),
            doc.widthAnchor.constraint(equalTo: scroll.contentView.widthAnchor),
        ])
        scrollHeight = scroll.heightAnchor.constraint(equalToConstant: 0)
        scrollHeight.isActive = true

        // thinking row
        let spark = label("✳︎", ui(14, .semibold), olive, wrap: false)
        thinking.setViews([spark, thinkingText], in: .leading)
        thinking.spacing = 8
        thinking.isHidden = true

        let rule = NSBox()
        rule.boxType = .custom
        rule.fillColor = hairline
        rule.borderWidth = 0
        rule.translatesAutoresizingMaskIntoConstraints = false
        rule.heightAnchor.constraint(equalToConstant: 1).isActive = true

        // input
        field.isBordered = false
        field.drawsBackground = false
        field.focusRingType = .none
        field.font = ui(14)
        field.textColor = ink
        field.delegate = self
        field.target = self
        field.action = #selector(submit)
        setPlaceholder("Ask Hobby to do something on this Mac…")

        // bottom bar: chips ........ send
        send.isBordered = false
        send.wantsLayer = true
        send.layer?.cornerRadius = 8
        send.attributedTitle = NSAttributedString(string: "↑", attributes: [.font: ui(13, .semibold), .foregroundColor: card])
        send.target = self
        send.action = #selector(submit)
        send.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([send.widthAnchor.constraint(equalToConstant: 26), send.heightAnchor.constraint(equalToConstant: 26)])
        updateSend()
        let gap = NSView()
        gap.setContentHuggingPriority(.init(1), for: .horizontal)
        let bottom = NSStackView(views: [
            Pill("HobbyLM-1B", font: ui(11.5, .medium), fill: chipFill),
            Pill("Offline · \(hotKeyName())", font: ui(11.5, .medium), fill: chipFill, color: muted),
            gap, send])
        bottom.spacing = 6
        bottom.alignment = .centerY

        let stack = NSStackView(views: [header, scroll, thinking, rule, field, bottom])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 12
        stack.edgeInsets = NSEdgeInsets(top: 15, left: 16, bottom: 12, right: 12)
        stack.translatesAutoresizingMaskIntoConstraints = false
        stack.setCustomSpacing(16, after: header)
        stack.setCustomSpacing(12, after: rule)
        box.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: box.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: box.trailingAnchor),
            stack.topAnchor.constraint(equalTo: box.topAnchor),
            stack.bottomAnchor.constraint(equalTo: box.bottomAnchor),
            box.widthAnchor.constraint(equalToConstant: panelWidth),
        ])
        for v in [header, scroll, thinking, rule, field, bottom] as [NSView] {
            v.translatesAutoresizingMaskIntoConstraints = false
            v.widthAnchor.constraint(equalTo: stack.widthAnchor, constant: -(16 + 12)).isActive = true
        }
        setTranscriptVisible(false)
    }

    func setTranscriptVisible(_ on: Bool) { resize() }

    @objc func clearChat() {
        guard mode == .ready || mode == .loading else { return }  // not while Hobby is working or waiting for a yes/no
        transcript.arrangedSubviews.forEach { $0.removeFromSuperview() }
        transcriptText = ""
        lastStep = nil
        setClearEnabled(false)
        resize()
        panel.makeFirstResponder(field)
    }

    func setClearEnabled(_ on: Bool) {
        clear.isEnabled = on
        clear.attributedTitle = NSAttributedString(string: "Clear", attributes: [
            .font: ui(11.5, .medium), .foregroundColor: on ? muted : rgb(0xC4C2BB)])
    }

    func toggleCollapsed() { collapsed.toggle(); resize() }

    func toggleTall() { tall.toggle(); if tall { collapsed = false }; resize() }

    /// Fit the panel to its content, keeping the top edge where it is.
    func resize() {
        panel.contentView?.layoutSubtreeIfNeeded()
        scroll.isHidden = collapsed || transcript.arrangedSubviews.isEmpty
        let docH = transcript.fittingSize.height
        scrollHeight.constant = scroll.isHidden ? 0 : min(docH, maxTranscript)
        panel.contentView?.layoutSubtreeIfNeeded()
        let h = panel.contentView?.fittingSize.height ?? 200
        var f = panel.frame
        f.origin.y += f.height - h
        f.size = NSSize(width: panelWidth, height: h)
        resizing = true
        panel.setFrame(f, display: true)
        resizing = false
        if docH > maxTranscript, let doc = scroll.documentView {
            doc.scroll(NSPoint(x: 0, y: docH - maxTranscript))
        }
    }

    func setPlaceholder(_ text: String) {
        field.placeholderAttributedString = NSAttributedString(string: text, attributes: [.foregroundColor: rgb(0x9A9A92), .font: ui(14)])
    }

    func controlTextDidChange(_ obj: Notification) { updateSend() }

    func updateSend() {
        let on = !field.stringValue.trimmingCharacters(in: .whitespaces).isEmpty && (mode == .ready || mode == .asking)
        send.layer?.backgroundColor = (on ? ink : line).cgColor
    }

    func setStatus(_ text: String, live: Bool) {
        status.stringValue = text
        statusDot.layer?.backgroundColor = (live ? lime : line).cgColor
    }

    func setThinking(_ on: Bool) {
        thinking.isHidden = !on
        dotTimer?.invalidate()
        if on {
            dots = 0
            dotTimer = Timer.scheduledTimer(withTimeInterval: 0.35, repeats: true) { [weak self] _ in
                guard let self else { return }
                self.dots = (self.dots + 1) % 4
                self.thinkingText.stringValue = "Thinking" + String(repeating: ".", count: self.dots)
            }
        }
        resize()
    }

    func show() {
        guard let screen = NSScreen.screens.first(where: { NSMouseInRect(NSEvent.mouseLocation, $0.frame, false) }) ?? NSScreen.main
            ?? NSScreen.screens.first else { return }
        let v = screen.visibleFrame
        resize()
        // reopen where the user last dragged it, if that spot is still on a screen; otherwise centred near the top
        if let saved = UserDefaults.standard.string(forKey: "hobbyBarTopLeft").map(NSPointFromString),
           NSScreen.screens.contains(where: { NSPointInRect(NSPoint(x: saved.x + 20, y: saved.y - 20), $0.visibleFrame) }) {
            panel.setFrameTopLeftPoint(saved)
        } else {
            panel.setFrameOrigin(NSPoint(x: v.midX - panelWidth / 2, y: v.maxY - v.height * 0.24 - panel.frame.height))
        }
        panel.makeKeyAndOrderFront(nil)
        panel.makeFirstResponder(field)
        field.currentEditor()?.selectAll(nil)
    }

    @objc func toggle() { panel.isVisible ? panel.orderOut(nil) : show() }

    @objc func resetPosition() {
        UserDefaults.standard.removeObject(forKey: "hobbyBarTopLeft")
        if panel.isVisible { show() }
    }

    // MARK: transcript rows

    func add(_ v: NSView, text: String) {
        transcript.addArrangedSubview(v)
        v.widthAnchor.constraint(lessThanOrEqualTo: transcript.widthAnchor, constant: -indent).isActive = true
        transcriptText += text + "\n"
        setClearEnabled(mode == .ready)
        resize()
    }

    func addText(_ s: String, _ font: NSFont = ui(13), _ color: NSColor = muted) {
        let l = label(s, font, color)
        l.preferredMaxLayoutWidth = panelWidth - 28 - indent
        add(l, text: s)
    }

    func addStep(seconds: Double?, action: String) {
        let chip = Pill(action, font: ui(12.5, .medium), fill: limeSoft, color: olive)
        let time = label(seconds.map { String(format: "%.1fs", $0) } ?? "", mono(10.5, .regular), rgb(0x9A9A92), wrap: false)
        time.setContentCompressionResistancePriority(.required, for: .horizontal)
        let row = NSStackView(views: [chip, time])
        row.spacing = 8
        row.alignment = .centerY
        lastStep = row
        add(row, text: "\(time.stringValue)  Hobby → \(action)")
    }

    func addResult(_ s: String) {
        let l = label(s, ui(13), ink, wrap: false)
        l.lineBreakMode = .byTruncatingTail
        l.toolTip = s
        l.setContentCompressionResistancePriority(.init(100), for: .horizontal)  // the result truncates, never the timing
        if let row = lastStep, row.arrangedSubviews.count == 2 {
            row.insertArrangedSubview(l, at: 1)  // chip · result · time
            transcriptText += "↳ \(s)\n"
            resize()
        } else {
            addText("↳ " + s)
        }
    }

    func addEnd(kind: String, text: String) {
        guard kind == "done" else { addText(text, ui(13), kind == "unknown" ? ink : muted); return }
        let check = Pill("✓", font: ui(10, .bold), fill: lime)
        let time = label(doneSeconds.map { String(format: "%.1fs", $0) } ?? "", mono(10.5, .regular), rgb(0x9A9A92), wrap: false)
        doneSeconds = nil
        let row = NSStackView(views: [check, label("Done", ui(13, .medium), ink, wrap: false), time])
        row.spacing = 7
        row.alignment = .centerY
        add(row, text: "✓ done")
    }

    func addConfirm(_ s: String) {
        addText(s, ui(13, .medium), ink)
    }

    func addChoice() {
        let yes = PillButton("Add event", filled: true)
        let no = PillButton("Cancel", filled: false)
        yes.onClick = { [weak self] in self?.answer("y", shown: "Add event") }
        no.onClick = { [weak self] in self?.answer("n", shown: "Cancel") }
        let row = NSStackView(views: [yes, no])
        row.spacing = 8
        add(row, text: "[Add event] [Cancel]")
    }

    // MARK: menu bar + hotkey

    func buildMenu() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        item.button?.title = "hobby"
        let menu = NSMenu()
        menu.addItem(withTitle: "Open Hobby (\(hotKeyName()))", action: #selector(toggle), keyEquivalent: "").target = self
        menu.addItem(withTitle: "Reset position", action: #selector(resetPosition), keyEquivalent: "").target = self
        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit Hobby", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        item.menu = menu
        statusItem = item
    }

    /// (key code, Carbon modifiers, label) for HOBBY_BAR_HOTKEY
    func hotKeySpec() -> (Int, Int, String) {
        switch ProcessInfo.processInfo.environment["HOBBY_BAR_HOTKEY"] {
        case "cmd-shift-h": return (kVK_ANSI_H, cmdKey | shiftKey, "⇧⌘H")
        case "ctrl-option-space": return (kVK_Space, controlKey | optionKey, "⌃⌥Space")
        case "option-space": return (kVK_Space, optionKey, "⌥Space")
        case "cmd-shift-space": return (kVK_Space, cmdKey | shiftKey, "⇧⌘Space")
        case "cmd-h": return (kVK_ANSI_H, cmdKey, "⌘H")
        default: return (kVK_ANSI_H, cmdKey | shiftKey, "⇧⌘H")
        }
    }

    func hotKeyName() -> String { hotKeySpec().2 }

    func registerHotKey() {
        let (key, mods, _) = hotKeySpec()
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        InstallEventHandler(GetApplicationEventTarget(), { _, _, _ in
            DispatchQueue.main.async { (NSApp.delegate as? Hobby)?.toggle() }
            return noErr
        }, 1, &spec, nil, nil)
        let id = EventHotKeyID(signature: OSType(0x4842_4152), id: 1)  // 'HBAR'
        let err = RegisterEventHotKey(UInt32(key), UInt32(mods), id, GetApplicationEventTarget(), 0, &hotKey)
        hotKeyStatus = err == noErr ? "\(hotKeyName()) registered" : "\(hotKeyName()) failed (OSStatus \(err))"
        if err != noErr {
            addText("Could not register \(hotKeyName()); another app may be using it. Use the menu bar item instead.")
        }
    }

    // MARK: worker

    func startWorker() {
        let python = root.appendingPathComponent(".venv/bin/python")
        let server = ProcessInfo.processInfo.environment["HOBBY_BAR_WORKER"].map { URL(fileURLWithPath: $0) }  // tests: a stand-in worker
            ?? root.appendingPathComponent("hobby/hobby_server.py")
        let p = Process()
        p.executableURL = python
        p.arguments = [server.path, "r4"]
        p.currentDirectoryURL = root
        var env = ProcessInfo.processInfo.environment
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["HOBBY_HOME"] = root.path
        p.environment = env
        let input = Pipe(), output = Pipe()
        p.standardInput = input
        p.standardOutput = output
        let logURL = root.appendingPathComponent("data/hobby-bar.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        p.standardError = try? FileHandle(forWritingTo: logURL)
        output.fileHandleForReading.readabilityHandler = { [weak self] h in
            let d = h.availableData
            if d.isEmpty { h.readabilityHandler = nil; return }  // EOF: the worker exited
            DispatchQueue.main.async { self?.receive(d) }
        }
        p.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async {
                guard let self else { return }
                self.mode = .stopped  // no more writes to the worker's closed stdin
                self.toWorker = nil
                self.setStatus("Stopped", live: false)
                self.setThinking(false)
                self.setPlaceholder("Hobby stopped. Quit and reopen it from the menu bar.")
                self.updateSend()
                self.addText("Hobby stopped. Details: data/hobby-bar.log")
                if self.selftest != nil {
                    self.sendToWorker(["type": "request", "text": "probe"])  // must be a no-op now, not a crash
                    self.finishSelftest()
                }
            }
        }
        do { try p.run() } catch {
            setStatus("Not started", live: false)
            addText("Could not start \(python.path). Run scripts/setup.sh first.")
            return
        }
        worker = p
        toWorker = input.fileHandleForWriting
    }

    func sendToWorker(_ obj: [String: String]) {
        guard mode != .stopped, let pipe = toWorker, let d = try? JSONSerialization.data(withJSONObject: obj) else { return }
        do { try pipe.write(contentsOf: d + Data("\n".utf8)) } catch {  // worker gone between checks: stop cleanly
            mode = .stopped
            toWorker = nil
            setStatus("Stopped", live: false)
            setThinking(false)
            addText("Hobby stopped. Details: data/hobby-bar.log")
        }
    }

    func receive(_ d: Data) {
        buffer.append(d)
        while let nl = buffer.firstIndex(of: 0x0A) {
            let lineData = buffer[buffer.startIndex..<nl]
            buffer.removeSubrange(buffer.startIndex...nl)
            if let msg = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any] { handle(msg) }
        }
    }

    func handle(_ m: [String: Any]) {
        guard mode != .stopped else { return }  // a leftover end/ready from a dead worker must not revive the panel
        let text = m["text"] as? String ?? ""
        switch m["type"] as? String {
        case "ready":
            mode = .ready
            setStatus("Offline · ready", live: true)
            updateSend()
            if (m["accessibility"] as? Bool) == false {
                addText("Accessibility is off for this app, so Calculator typing will fail. Turn it on in System Settings → Privacy & Security → Accessibility.")
            }
            if selftest != nil {
                show()
                selftestNext()
            }
        case "step" where (m["action"] as? String) == "DONE":
            doneSeconds = m["seconds"] as? Double  // shown on the Done row instead of its own chip
        case "step":
            addStep(seconds: m["seconds"] as? Double, action: m["action"] as? String ?? "")
        case "result": addResult(text)
        case "info", "warn", "error": addText(text)
        case "confirm": addConfirm(text)
        case "ask" where selftest != nil:  // the self-test never confirms anything
            addText("Cancel (self-test)")
            sendToWorker(["type": "answer", "text": "n"])
        case "ask":
            mode = .asking
            setClearEnabled(false)
            setThinking(false)
            setStatus("Waiting for you", live: true)
            let prompt = m["prompt"] as? String ?? "answer"
            if prompt.hasPrefix("y / n") { addChoice(); setPlaceholder("Type y or n, or use the buttons") } else { setPlaceholder(prompt) }
            field.stringValue = ""
            updateSend()
            panel.makeFirstResponder(field)
        case "end":
            mode = .ready
            setThinking(false)
            setStatus("Offline · ready", live: true)
            setPlaceholder("Ask Hobby to do something on this Mac…")
            addEnd(kind: m["kind"] as? String ?? "", text: text)
            field.stringValue = ""
            updateSend()
            if panel.isVisible { panel.makeKeyAndOrderFront(nil) }  // an app Hobby opened (Finder, Notes) took focus
            panel.makeFirstResponder(field)
            if selftest != nil {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) { self.selftestQueue.isEmpty ? self.finishSelftest() : self.selftestNext() }
            }
        default: break
        }
    }

    // MARK: input

    @objc func submit() {
        let text = field.stringValue.trimmingCharacters(in: .whitespaces)
        switch mode {
        case .ready where !text.isEmpty:
            if let last = transcript.arrangedSubviews.last { transcript.setCustomSpacing(16, after: last) }  // gap between requests
            lastStep = nil
            addText(text, ui(13), muted)
            sendToWorker(["type": "request", "text": text])
            mode = .busy
            setClearEnabled(false)
            setStatus("Working", live: true)
            setThinking(true)
            field.stringValue = ""
        case .asking:
            answer(text, shown: text.isEmpty ? "(no answer)" : text)
        default: break
        }
        updateSend()
    }

    func answer(_ text: String, shown: String) {
        guard mode == .asking else { return }
        addText("→ " + shown, ui(13, .medium), ink)
        sendToWorker(["type": "answer", "text": text])
        mode = .busy
        setClearEnabled(false)
        setStatus("Working", live: true)
        setThinking(true)
        field.stringValue = ""
        setPlaceholder("Ask Hobby to do something on this Mac…")
        updateSend()
    }

    lazy var selftestQueue: [String] = (selftest ?? "").components(separatedBy: "||").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }

    func selftestNext() {
        field.stringValue = selftestQueue.removeFirst()
        submit()
    }

    func finishSelftest() {
        let out = root.appendingPathComponent("data/hobby-bar-selftest")
        let info = "hotkey: \(hotKeyStatus)\npanel visible: \(panel.isVisible), key: \(panel.isKeyWindow)\n"
            + "frontmost app: \(NSWorkspace.shared.frontmostApplication?.localizedName ?? "?")\n---\n" + transcriptText
        try? info.write(to: out.appendingPathExtension("txt"), atomically: true, encoding: .utf8)
        let shot = Process()  // -l: this window only, nothing else on screen
        shot.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
        shot.arguments = ["-x", "-o", "-l", String(panel.windowNumber), out.appendingPathExtension("png").path]
        try? shot.run()
        shot.waitUntilExit()
        NSApp.terminate(nil)
    }

    func applicationWillTerminate(_ note: Notification) {
        try? toWorker?.close()
        worker?.terminate()
    }
}

// The repository root: HOBBY_HOME, else two levels up from data/HobbyBar.
let exe = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
let root = ProcessInfo.processInfo.environment["HOBBY_HOME"].map { URL(fileURLWithPath: $0) }
    ?? exe.deletingLastPathComponent().deletingLastPathComponent()
signal(SIGPIPE, SIG_IGN)  // a write to a dead worker becomes a catchable error, not a crash
let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let delegate = Hobby(root: root)
app.delegate = delegate
app.run()

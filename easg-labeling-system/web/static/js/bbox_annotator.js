(function() {
    // Use coffee-script compiler to obtain a javascript file.

    //    coffee -c bbox_annotator.coffee

    // See http://coffeescript.org/

    // BBox selection window.
    var BBoxSelector;

    BBoxSelector = class BBoxSelector {
    // Initializes selector in the image frame.
    constructor(image_frame, options) {
        if (options == null) {
        options = {};
        }
        options.input_method || (options.input_method = "text");
        this.image_frame = image_frame;
        this.border_width = options.border_width || 2;
        this.selector = $('<div class="bbox_selector"></div>');
        this.selector.css({
            "border": this.border_width + "px dotted rgb(127,255,127)",
            "position": "absolute"
        });
        this.image_frame.append(this.selector);
        this.selector.css({
            "border-width": this.border_width
        });
        this.selector.hide();
        this.create_label_box(options);
    }

    // Initializes a label input box.
    create_label_box(options) {
        var i, label, len, ref;
        options.labels || (options.labels = ["object"]);
        this.label_box = $('<div class="label_box"></div>');
        this.label_box.css({
        "position": "absolute"
        });
        this.image_frame.append(this.label_box);
        switch (options.input_method) {
        case 'select':
            if (typeof options.labels === "string") {
            options.labels = [options.labels];
            }
            this.label_input = $('<select class="label_input" name="label"></select>');
            this.label_box.append(this.label_input);
            this.label_input.append($('<option value>choose an item</option>'));
            ref = options.labels;
            for (i = 0, len = ref.length; i < len; i++) {
            label = ref[i];
            this.label_input.append('<option value="' + label + '">' + label + '</option>');
            }
            this.label_input.change(function(e) {
            return this.blur();
            });
            break;
        case 'text':
            if (typeof options.labels === "string") {
            options.labels = [options.labels];
            }
            this.label_input = $('<input class="label_input" name="label" ' + 'type="text" value>');
            this.label_box.append(this.label_input);
            this.label_input.autocomplete({
            source: options.labels || [''],
            autoFocus: true
            });
            break;
        case 'fixed':
            if ($.isArray(options.labels)) {
            options.labels = options.labels[0];
            }
            this.label_input = $('<input class="label_input" name="label" type="text">');
            this.label_box.append(this.label_input);
            this.label_input.val(options.labels);
            break;
        default:
            throw 'Invalid label_input parameter: ' + options.input_method;
        }
        return this.label_box.hide();
    }

    // Crop x and y to the image size.
    crop(pageX, pageY) {
        var point;
        return point = {
        x: Math.min(Math.max(Math.round(pageX - this.image_frame.offset().left), 0), Math.round(this.image_frame.width() - 1)),
        y: Math.min(Math.max(Math.round(pageY - this.image_frame.offset().top), 0), Math.round(this.image_frame.height() - 1))
        };
    }

    // When a new selection is made.
    start(pageX, pageY) {
        this.pointer = this.crop(pageX, pageY);
        this.offset = this.pointer;
        this.refresh();
        this.selector.show();
        $('body').css('cursor', 'crosshair');
        return document.onselectstart = function() {
        return false;
        };
    }

    // When a new selection is made.
    startResize(X, Y) {
        this.pointer = {x:X,y:Y};
        this.offset = this.pointer;
        this.refresh();
        this.selector.show();
        $('body').css('cursor', 'crosshair');
        return document.onselectstart = function() {
        return false;
        };
    }

    // When a selection updates.
    update_rectangle(pageX, pageY) {
        this.pointer = this.crop(pageX, pageY);
        return this.refresh();
    }

    // When starting to input label.
    input_label(options) {
        $('body').css('cursor', 'default');
        document.onselectstart = function() {
        return true;
        };
        this.label_box.show();
        return this.label_input.focus();
    }

    // Finish and return the annotation.
    finish(options) {
        var data;
        this.label_box.hide();
        this.selector.hide();
        data = this.rectangle();
        data.label = $.trim(this.label_input.val().toLowerCase());
        if (options.input_method !== 'fixed') {
        this.label_input.val('');
        }
        return data;
    }

    // Get a rectangle.
    rectangle() {
        var rect, x1, x2, y1, y2;
        x1 = Math.min(this.offset.x, this.pointer.x);
        y1 = Math.min(this.offset.y, this.pointer.y);
        x2 = Math.max(this.offset.x, this.pointer.x);
        y2 = Math.max(this.offset.y, this.pointer.y);
        return rect = {
        left: x1,
        top: y1,
        width: x2 - x1 + 1,
        height: y2 - y1 + 1
        };
    }

    // edit_rectangle(editPoint) {
    //     var data;
    //     data = this.rectangle();
    //     console.log(data);
    //     console.log("sono nella funzione di edit_rectangle");
    //     switch (editPoint) {
    //         case 0:
                
    //             console.log(editPoint);
    //             break;
    //         case 1: 
    //             console.log(editPoint);
    //             break;

    //         case 2:
    //             console.log(editPoint);
    //             break;

    //         case 3:
    //             console.log(editPoint);
    //             break;

    //     }
    // }

    // Update css of the box.
    refresh() {
        var rect;
        rect = this.rectangle();
        this.selector.css({
        left: (rect.left - this.border_width) + 'px',
        top: (rect.top - this.border_width) + 'px',
        width: rect.width + 'px',
        height: rect.height + 'px'
        });
        return this.label_box.css({
        left: (rect.left - this.border_width) + 'px',
        top: (rect.top + rect.height + this.border_width) + 'px'
        });
    }

    // Return input element.
    get_input_element() {
        return this.label_input;
    }

    };

    // Annotator object definition.
    this.BBoxAnnotator = class BBoxAnnotator {
        // Initialize the annotator layout and events.
        constructor(options) {
            var annotator, image_element;
            annotator = this;
            this.annotator_element = $(options.id || "#bbox_annotator");
            this.border_width = options.border_width || 2;
            this.show_label = options.show_label || (options.input_method !== "fixed");
            if (options.multiple != null) {
                this.multiple = options.multiple;
            } else {
                this.multiple = true;
            }
            this.image_frame = $('<div class="image_frame"></div>');
            this.annotator_element.append(this.image_frame);
            if (options.guide) {
                annotator.initialize_guide(options.guide);
            }
            image_element = new Image();
            image_element.src = options.url;
            image_element.onload = function() {
                if (options.width) {
                    let aspect_ratio = image_element.width / image_element.height;
                    options.height = Math.round(options.width / aspect_ratio);
                } else if (options.height) {
                    let aspect_ratio = image_element.width / image_element.height;
                    options.width = Math.round(options.height * aspect_ratio);
                }
                else {
                    options.width = image_element.width;
                    options.height = image_element.height;
                }
                annotator.annotator_element.css({
                    "width": (options.width + annotator.border_width) + 'px',
                    "height": (options.height + annotator.border_width) + 'px',
                    "padding-left": (annotator.border_width / 2) + 'px',
                    "padding-top": (annotator.border_width / 2) + 'px',
                    "cursor": "crosshair",
                    "overflow": "hidden"
                });
                annotator.image_frame.css({
                    "background-image": "url('" + image_element.src + "')",
                    "width": options.width + "px",
                    "height": options.height + "px",
                    "position": "relative"
                });
                annotator.selector = new BBoxSelector(annotator.image_frame, options);
                return annotator.initialize_events(options);
            };
            image_element.onerror = function() {
                return annotator.annotator_element.text("Invalid image URL: " + options.url);
            };
            this.entries = [];
            this.onchange = options.onchange;
        }
    
        // Initialize events.
        initialize_events(options) {
            var annotator, selector, status;
            status = 'free';
            this.hit_menuitem = false;
            annotator = this;
            selector = annotator.selector;

            this.annotator_element.mousedown(function(e) {
                
                if (!annotator.hit_menuitem) {
                    switch (status) {
                    case 'free':
                    case 'input':
                        if (status === 'input') {
                        selector.get_input_element().blur();
                        }
                        if (e.which === 1) { // left button
                            if (annotator.entries.length == 0){
                                selector.start(e.pageX, e.pageY);
                                status = 'hold';
                            }
                        }
                    }
                }
                annotator.hit_menuitem = false;
                return true;
            });
            $(window).mousemove(function(e) {
                
                var offset;
                switch (status) {
                    case 'hold':
                        if (annotator.entries.length == 0){
                            selector.update_rectangle(e.pageX, e.pageY);
                        }
                }
                if (annotator.guide_h) {
                    offset = annotator.image_frame.offset();
                    annotator.guide_h.css('top', Math.floor(e.pageY - offset.top) + 'px');
                    annotator.guide_v.css('left', Math.floor(e.pageX - offset.left) + 'px');
                }
                return true;
            });
            $(window).mouseup(function(e) {
                
                switch (status) {
                    case 'hold':
                        if (annotator.entries.length == 0){
                            selector.update_rectangle(e.pageX, e.pageY);
                            selector.input_label(options);
                            status = 'input';
                            if (options.input_method === 'fixed') {
                                selector.get_input_element().blur();
                            }
                        }
                }
                return true;
            });
            selector.get_input_element().blur(function(e) {
            var data;
            switch (status) {
                case 'input':
                data = selector.finish(options);
                if (data.label) {
                    annotator.add_entry(data);
                    if (annotator.onchange) {
                    annotator.onchange(annotator.entries);
                    }
                }
                status = 'free';
            }
            return true;
            });
            selector.get_input_element().keypress(function(e) {
                switch (status) {
                    case 'input':
                    if (e.which === 13) {
                        selector.get_input_element().blur();
                    }
                }
                return e.which !== 13;
            });
            selector.get_input_element().mousedown(function(e) {
                return annotator.hit_menuitem = true;
            });
            selector.get_input_element().mousemove(function(e) {
                return annotator.hit_menuitem = true;
            });
            selector.get_input_element().mouseup(function(e) {
                return annotator.hit_menuitem = true;
            });
            return selector.get_input_element().parent().mousedown(function(e) {
                return annotator.hit_menuitem = true;
            });
        }
    
        // Add a new entry.
        add_entry(entry) {
            var annotator, box_element, close_button, text_box;
            var edit_button_top, edit_button_bottom, edit_button_left, edit_button_right;
            var resize_status = 'free';
            if (!this.multiple) {
                this.annotator_element.find(".annotated_bounding_box").detach();
                this.entries.splice(0);
            }
            this.entries.push(entry);
            box_element = $('<div class="annotated_bounding_box"></div>');
            box_element.appendTo(this.image_frame).css({
                "border": this.border_width + "px solid rgb(127,255,127)",
                "position": "absolute",
                "top": (entry.top - this.border_width) + "px",
                "left": (entry.left - this.border_width) + "px",
                "width": entry.width + "px",
                "height": entry.height + "px",
                "color": "rgb(127,255,127)",
                "font-family": "monospace",
                "font-size": "small"
            });
            close_button = $('<div></div>').appendTo(box_element).css({
                "position": "absolute",
                "top": "-8px",
                "right": "-8px",
                "width": "16px",
                "height": "0",
                "padding": "16px 0 0 0",
                "overflow": "hidden",
                "color": "#fff",
                "background-color": "#030",
                "border": "2px solid #fff",
                "-moz-border-radius": "18px",
                "-webkit-border-radius": "18px",
                "border-radius": "18px",
                "cursor": "pointer",
                "-moz-user-select": "none",
                "-webkit-user-select": "none",
                "user-select": "none",
                "text-align": "center"
            });
            $("<div></div>").appendTo(close_button).html('X').css({
                "display": "block",
                "text-align": "center",
                "width": "16px",
                "position": "absolute",
                "top": "-2px",
                "left": "0",
                "font-size": "16px",
                "line-height": "16px",
                "font-family": '"Helvetica Neue", Consolas, Verdana, Tahoma, Calibri, ' + 'Helvetica, Menlo, "Droid Sans", sans-serif'
            });

            edit_button_top = $('<button class="edit-button-top"></button>').appendTo(box_element).css({
                "position": "absolute",
                "top": "-4px",
                "left": "50%",
                "width": "6px",
                "height": "6px",
                "padding": "0 0 0 0",
                "overflow": "hidden",
                "color": "#fff",
                "background-color": "#030",
                "border": "1px solid #fff",
                "-moz-border-radius": "1px",
                "-webkit-border-radius": "1px",
                "border-radius": "1px",
                "cursor": "pointer",
                "-moz-user-select": "none",
                "-webkit-user-select": "none",
                "user-select": "none",
                "text-align": "center"
            });
            edit_button_bottom = $('<button class="edit-button-bottom"></button>').appendTo(box_element).css({
                "position": "absolute",
                "top": "98%",
                "left": "50%",
                "width": "6px",
                "height": "6px",
                "padding": "0 0 0 0",
                "overflow": "hidden",
                "color": "#fff",
                "background-color": "#030",
                "border": "1px solid #fff",
                "-moz-border-radius": "1px",
                "-webkit-border-radius": "1px",
                "border-radius": "1px",
                "cursor": "pointer",
                "-moz-user-select": "none",
                "-webkit-user-select": "none",
                "user-select": "none",
                "text-align": "center"
            });
            edit_button_left = $('<button class="edit-button-left"></button>').appendTo(box_element).css({
                "position": "absolute",
                "top": "50%",
                "left": "-5px",
                "width": "6px",
                "height": "6px",
                "padding": "0 0 0 0",
                "overflow": "hidden",
                "color": "#fff",
                "background-color": "#030",
                "border": "1px solid #fff",
                "-moz-border-radius": "1px",
                "-webkit-border-radius": "1px",
                "border-radius": "1px",
                "cursor": "pointer",
                "-moz-user-select": "none",
                "-webkit-user-select": "none",
                "user-select": "none",
                "text-align": "center"
            });
            edit_button_right = $('<button class="edit-button-right"></button>').appendTo(box_element).css({
                "position": "absolute",
                "top": "50%",
                "left": "99%",
                "width": "6px",
                "height": "6px",
                "padding": "0 0 0 0",
                "overflow": "hidden",
                "color": "#fff",
                "background-color": "#030",
                "border": "1px solid #fff",
                "-moz-border-radius": "1px",
                "-webkit-border-radius": "1px",
                "border-radius": "1px",
                "cursor": "pointer",
                "-moz-user-select": "none",
                "-webkit-user-select": "none",
                "user-select": "none",
                "text-align": "center"
            });

            text_box = $('<div></div>').appendTo(box_element).css({
                "overflow": "hidden"
            });
            if (this.show_label) {
                text_box.text(entry.label);
            }
            annotator = this;
            box_element.hover((function(e) {
                return close_button.show();
            }), (function(e) {
                return close_button.hide();
            }));
            close_button.mousedown(function(e) {
                return annotator.hit_menuitem = true;
            });
            close_button.click(function(e) {
                var clicked_box, index;
                clicked_box = close_button.parent(".annotated_bounding_box");
                index = clicked_box.prevAll(".annotated_bounding_box").length;
                clicked_box.detach();
                annotator.entries.splice(index, 1);
                return annotator.onchange(annotator.entries);
            });

            edit_button_top.mousedown(function(e) {
                if (JSON.stringify(resize_status) === JSON.stringify('free')){
                    resize_status = 'hold_top';
                    var clicked_box = edit_button_top.parent(".annotated_bounding_box")[0];
                    clicked_box.style.border = "2px dotted rgb(127, 255, 127)";
                }
              });
              edit_button_left.mousedown(function(e) {
                if (JSON.stringify(resize_status) === JSON.stringify('free')){
                    resize_status = 'hold_left';
                    var clicked_box = edit_button_left.parent(".annotated_bounding_box")[0];
                    clicked_box.style.border = "2px dotted rgb(127, 255, 127)";
                }
              });
              edit_button_right.mousedown(function(e) {
                if (JSON.stringify(resize_status) === JSON.stringify('free')){
                    resize_status = 'hold_right';
                    var clicked_box = edit_button_right.parent(".annotated_bounding_box")[0];
                    clicked_box.style.border = "2px dotted rgb(127, 255, 127)";
                }
              });
              edit_button_bottom.mousedown(function(e) {
                if (JSON.stringify(resize_status) === JSON.stringify('free')){
                    resize_status = 'hold_bottom';
                    var clicked_box = edit_button_bottom.parent(".annotated_bounding_box")[0];
                    clicked_box.style.border = "2px dotted rgb(127, 255, 127)";
                }
              });
              $(window).mousemove(function(e) {
    
                if (JSON.stringify(resize_status) === JSON.stringify('hold_top')){
                    var clicked_box = edit_button_top.parent(".annotated_bounding_box")[0];
                    annotator.selector.pointer = annotator.selector.crop(e.pageX, e.pageY);
                    var prevTop = parseInt(clicked_box.style.top.replace("px", ""));
                    var prevHeight = parseInt(clicked_box.style.height.replace("px", ""));
                    var offsetY; 
                    if (prevTop >= annotator.selector.pointer.y) {
                        clicked_box.style.top = annotator.selector.pointer.y;
                        offsetY = prevTop - annotator.selector.pointer.y;
                        clicked_box.style.height = prevHeight + offsetY;
                    }
                    else {
                        clicked_box.style.top = annotator.selector.pointer.y;
                        offsetY = annotator.selector.pointer.y - prevTop;
                        clicked_box.style.height = prevHeight - offsetY;
                    }
                }
                else if (JSON.stringify(resize_status) === JSON.stringify('hold_left')){
                    var clicked_box = edit_button_left.parent(".annotated_bounding_box")[0];
                    annotator.selector.pointer = annotator.selector.crop(e.pageX, e.pageY);
                    var prevLeft = parseInt(clicked_box.style.left.replace("px", ""));
                    var prevWidth = parseInt(clicked_box.style.width.replace("px", ""));
                    var offsetX; 
                    if (prevLeft >= annotator.selector.pointer.x) {
                        clicked_box.style.left = annotator.selector.pointer.x;
                        offsetX = prevLeft - annotator.selector.pointer.x;
                        clicked_box.style.width = prevWidth + offsetX;
                    }
                    else {
                        clicked_box.style.left = annotator.selector.pointer.x;
                        offsetX = annotator.selector.pointer.x - prevLeft;
                        clicked_box.style.width = prevWidth - offsetX;
                    }
                }
                else if (JSON.stringify(resize_status) === JSON.stringify('hold_right')){
                    var clicked_box = edit_button_right.parent(".annotated_bounding_box")[0];
                    annotator.selector.pointer = annotator.selector.crop(e.pageX, e.pageY);
                    var prevLeft = parseInt(clicked_box.style.left.replace("px", ""));
                    
                    if (annotator.selector.pointer.x >= prevLeft) {
                        console.log(JSON.stringify(annotator.selector.pointer.x - prevLeft)+""+"px");
                        clicked_box.style.width = JSON.stringify(annotator.selector.pointer.x - prevLeft)+""+"px";
                    }
                }
                else if (JSON.stringify(resize_status) === JSON.stringify('hold_bottom')){
                    var clicked_box = edit_button_top.parent(".annotated_bounding_box")[0];
                    annotator.selector.pointer = annotator.selector.crop(e.pageX, e.pageY);
                    var prevTop = parseInt(clicked_box.style.top.replace("px", ""));
                    var prevHeight = parseInt(clicked_box.style.height.replace("px", ""));
                    if (annotator.selector.pointer.y >= prevTop) {
                        clicked_box.style.height = annotator.selector.pointer.y - prevTop;
                    }
                }
              });
            //   edit_button_left.mousemove(function(e) {
            //     if (JSON.stringify(resize_status) === JSON.stringify('hold')){
            //         var clicked_box = edit_button_left.parent(".annotated_bounding_box")[0];
            //         annotator.selector.pointer = annotator.selector.crop(e.pageX, e.pageY);
            //         var prevLeft = parseInt(clicked_box.style.left.replace("px", ""));
            //         var prevWidth = parseInt(clicked_box.style.width.replace("px", ""));
            //         var offsetX; 
            //         if (prevLeft >= annotator.selector.pointer.x) {
            //             clicked_box.style.left = annotator.selector.pointer.x;
            //             offsetX = prevLeft - annotator.selector.pointer.x;
            //             clicked_box.style.width = prevWidth + offsetX;
            //         }
            //         else {
            //             clicked_box.style.left = annotator.selector.pointer.x;
            //             offsetX = annotator.selector.pointer.x - prevLeft;
            //             clicked_box.style.width = prevWidth - offsetX;
            //         }
            //     }
            //   });
            //   edit_button_right.mousemove(function(e) {
            //     if (JSON.stringify(resize_status) === JSON.stringify('hold')){
            //         var clicked_box = edit_button_right.parent(".annotated_bounding_box")[0];
            //         annotator.selector.pointer = annotator.selector.crop(e.pageX, e.pageY);
            //         var prevLeft = parseInt(clicked_box.style.left.replace("px", ""));
                    
            //         if (annotator.selector.pointer.x >= prevLeft) {
            //             console.log(JSON.stringify(annotator.selector.pointer.x - prevLeft)+""+"px");
            //             clicked_box.style.width = JSON.stringify(annotator.selector.pointer.x - prevLeft)+""+"px";
            //         }
            //     }
            //   });
            //   edit_button_bottom.mousemove(function(e) {
            //     if (JSON.stringify(resize_status) === JSON.stringify('hold')){
            //         var clicked_box = edit_button_top.parent(".annotated_bounding_box")[0];
            //         annotator.selector.pointer = annotator.selector.crop(e.pageX, e.pageY);
            //         var prevTop = parseInt(clicked_box.style.top.replace("px", ""));
            //         var prevHeight = parseInt(clicked_box.style.height.replace("px", ""));
            //         if (annotator.selector.pointer.y >= prevTop) {
            //             clicked_box.style.height = annotator.selector.pointer.y - prevTop;
            //         }
            //     }
            //   });
              $(window).mouseup(function(e) {
                var clicked_box = edit_button_top.parent(".annotated_bounding_box")[0];
                clicked_box.style.border = "2px solid rgb(127, 255, 127)";
                resize_status = 'free';
                var newTop = parseInt(clicked_box.style.top.replace("px", ""));
                var newLeft = parseInt(clicked_box.style.left.replace("px", ""));
                var newWidth = parseInt(clicked_box.style.width.replace("px", ""));
                var newHeight = parseInt(clicked_box.style.height.replace("px", ""));
                if (annotator.entries.length == 1){
                    annotator.entries[0].top = newTop;
                    annotator.entries[0].left = newLeft;
                    annotator.entries[0].width = newWidth;
                    annotator.entries[0].height = newHeight;    
                }
                return annotator.onchange(annotator.entries);
              });
            //   edit_button_left.mouseup(function(e) {
            //     var clicked_box = edit_button_left.parent(".annotated_bounding_box")[0];
            //     clicked_box.style.border = "2px solid rgb(127, 255, 127)";
            //     resize_status = 'free';
            //     var newTop = parseInt(clicked_box.style.top.replace("px", ""));
            //     var newLeft = parseInt(clicked_box.style.left.replace("px", ""));
            //     var newWidth = parseInt(clicked_box.style.width.replace("px", ""));
            //     var newHeight = parseInt(clicked_box.style.height.replace("px", ""));
            //     if (annotator.entries.length == 1){
            //         annotator.entries[0].top = newTop;
            //         annotator.entries[0].left = newLeft;
            //         annotator.entries[0].width = newWidth;
            //         annotator.entries[0].height = newHeight;    
            //     }
            //     return annotator.onchange(annotator.entries);
            //   });
            //   edit_button_right.mouseup(function(e) {
            //     var clicked_box = edit_button_left.parent(".annotated_bounding_box")[0];
            //     clicked_box.style.border = "2px solid rgb(127, 255, 127)";
            //     resize_status = 'free';
            //     var newTop = parseInt(clicked_box.style.top.replace("px", ""));
            //     var newLeft = parseInt(clicked_box.style.left.replace("px", ""));
            //     var newWidth = parseInt(clicked_box.style.width.replace("px", ""));
            //     var newHeight = parseInt(clicked_box.style.height.replace("px", ""));
            //     if (annotator.entries.length == 1){
            //         annotator.entries[0].top = newTop;
            //         annotator.entries[0].left = newLeft;
            //         annotator.entries[0].width = newWidth;
            //         annotator.entries[0].height = newHeight;    
            //     }
            //     return annotator.onchange(annotator.entries);
            //   });
            //   edit_button_bottom.mouseup(function(e) {
            //     var clicked_box = edit_button_left.parent(".annotated_bounding_box")[0];
            //     clicked_box.style.border = "2px solid rgb(127, 255, 127)";
            //     resize_status = 'free';
            //     var newTop = parseInt(clicked_box.style.top.replace("px", ""));
            //     var newLeft = parseInt(clicked_box.style.left.replace("px", ""));
            //     var newWidth = parseInt(clicked_box.style.width.replace("px", ""));
            //     var newHeight = parseInt(clicked_box.style.height.replace("px", ""));
            //     if (annotator.entries.length == 1){
            //         annotator.entries[0].top = newTop;
            //         annotator.entries[0].left = newLeft;
            //         annotator.entries[0].width = newWidth;
            //         annotator.entries[0].height = newHeight;    
            //     }
            //     return annotator.onchange(annotator.entries);
            //   });
            return close_button.hide();
        }
    
        // Clear all entries.
        clear_all(e) {
            this.annotator_element.find(".annotated_bounding_box").detach();
            this.entries.splice(0);
            return this.onchange(this.entries);
        }
    
        // Add crosshair guide.
        initialize_guide(options) {
            this.guide_h = $('<div class="guide_h"></div>').appendTo(this.image_frame).css({
            "border": "1px dotted " + (options.color || '#000'),
            "height": "0",
            "width": "100%",
            "position": "absolute",
            "top": "0",
            "left": "0"
            });
            return this.guide_v = $('<div class="guide_v"></div>').appendTo(this.image_frame).css({
            "border": "1px dotted " + (options.color || '#000'),
            "height": "100%",
            "width": "0",
            "position": "absolute",
            "top": "0",
            "left": "0"
            });
        }
    };
}).call(this);
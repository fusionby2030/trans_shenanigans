module raylib
  use iso_c_binding, only: c_int32_t, c_char, c_int, c_bool, c_float, c_ptr
  use raymath
  implicit none

  ! typedef struct Texture {
  !   unsigned int id;        // OpenGL texture id
  !   int width;              // Texture base width
  !   int height;             // Texture base height
  !   int mipmaps;            // Mipmap levels, 1 by default
  !   int format;             // Data format (PixelFormat type)
  ! } Texture;
  type, bind(C) :: Texture
     integer(c_int) :: id, width, height, mipmap, format
  end type Texture

  ! typedef struct Font {
  !     int baseSize;           // Base size (default chars height)
  !     int glyphCount;         // Number of glyph characters
  !     int glyphPadding;       // Padding around the glyph characters
  !     Texture2D texture;      // Texture atlas containing the glyphs
  !     Rectangle *recs;        // Rectangles in texture for the glyphs
  !     GlyphInfo *glyphs;      // Glyphs info data
  ! } Font;
  type, bind(C) :: Font
     integer(c_int) :: baseSize, glyphCount, glyphPadding
     type(Texture) :: texture;
     type(c_ptr) :: recs, glyps;
  end type Font
  ! typedef struct Image {
  !     void *data;             // Image raw data
  !     int width;              // Image base width
  !     int height;             // Image base height
  !     int mipmaps;            // Mipmap levels, 1 by default
  !     int format;             // Data format (PixelFormat type)
  ! } Image;

  type, bind(C) :: Image
     type(c_ptr) :: data
     integer(c_int) :: width, height, mipmaps, fmt
  end type Image

  ! // Rectangle, 4 components
  ! typedef struct Rectangle {
  !     float x;                // Rectangle top-left corner position x
  !     float y;                // Rectangle top-left corner position y
  !     float width;            // Rectangle width
  !     float height;           // Rectangle height
  ! } Rectangle;
  type, bind(C) :: Rectangle
     real(c_float) :: x, y, width, height
  end type Rectangle

  ! typedef struct RenderTexture {
  !   unsigned int id;        // OpenGL framebuffer object id
  !   Texture texture;        // Color buffer attachment texture
  !   Texture depth;          // Depth buffer attachment texture
  ! } RenderTexture;
  type, bind(C) :: RenderTexture
     integer(c_int32_t) :: id
     type(Texture) :: texture, depth
  end type RenderTexture

  ! typedef struct Shader {
  !   unsigned int id;        // Shader program id
  !   int *locs;              // Shader locations array (RL_MAX_SHADER_LOCATIONS)
  ! } Shader;
  type, bind(C) :: Shader
     integer(c_int32_t) :: id
     type(c_ptr) :: locs
  end type Shader

  ! // Camera2D, defines position/orientation in 2d space
  ! typedef struct Camera2D {
  !     Vector2 offset;         // Camera offset (displacement from target)
  !     Vector2 target;         // Camera target (rotation and zoom origin)
  !     float rotation;         // Camera rotation in degrees
  !     float zoom;             // Camera zoom (scaling), should be 1.0f by default
  ! } Camera2D;
  type, bind(C) :: Camera2D
     type(Vector2) :: offset, target
     real(c_float) :: rotation, zoom
  end type Camera2D

  ! typedef struct AudioStream {
  !   rAudioBuffer *buffer;       // Pointer to internal data used by the audio system
  !   rAudioProcessor *processor; // Pointer to internal data processor, useful for audio effects
  !
  !   unsigned int sampleRate;    // Frequency (samples per second)
  !   unsigned int sampleSize;    // Bit depth (bits per sample): 8, 16, 32 (24 not supported)
  !   unsigned int channels;      // Number of channels (1-mono, 2-stereo, ...)
  ! } AudioStream;
  type, bind(C) :: AudioStream
     type(c_ptr) :: rAudioBuffer, rAudioProcessor
     integer(c_int32_t) :: sampleRate, sampleSize, channels
  end type AudioStream

  ! typedef struct Sound {
  !   AudioStream stream;         // Audio stream
  !   unsigned int frameCount;    // Total number of frames (considering channels)
  ! } Sound;
  type, bind(C) :: Sound
     type(AudioStream) :: stream
     integer(c_int32_t) :: frameCount
  end type Sound

  integer(c_int32_t), parameter :: BLANK = 0
  integer(c_int32_t), parameter :: BLACK = int(z'FF000000', c_int32_t)
  integer(c_int32_t), parameter :: WHITE = int(z'FFFFFFFF', c_int32_t)
  ! #define RED        CLITERAL(Color){ 230, 41, 55, 255 }     // Red
  integer(c_int32_t), parameter :: RED   = int(z'FF3729E6', c_int32_t)
  ! #define GREEN      CLITERAL(Color){ 0, 228, 48, 255 }      // Green
  integer(c_int32_t), parameter :: GREEN = int(z'FF30E400', c_int32_t)
  ! #define BLUE       CLITERAL(Color){ 0, 121, 241, 255 }     // Blue
  integer(c_int32_t), parameter :: BLUE  = int(z'FFF17900', c_int32_t)
  INTEGER(C_INT32_T), PARAMETER :: RAYWHITE = int(Z'FFF5F5F5', C_INT32_T)
  integer(c_int32_t), parameter :: NOVABLACK = int(z'FF3C4C55', c_int32_t)
  integer(c_int32_t), parameter :: NOVARED   = int(z'FFDF8C8C', c_int32_t)
  integer(c_int32_t), parameter :: NOVAGREEN = int(z'FFA8CE93', c_int32_t)
  integer(c_int32_t), parameter :: NOVABLUE  = int(z'FF83AFE5', c_int32_t)
  integer(c_int32_t), parameter :: MOUSE_BUTTON_LEFT = 0
  integer(c_int32_t), parameter :: KEY_P = 80
  integer(c_int32_t), parameter :: FLAG_WINDOW_RESIZABLE = int(z'00000004', c_int32_t)
  integer(c_int32_t), parameter :: FLAG_MSAA_4X_HINT     = int(z'00000020', c_int32_t)

  enum, bind(C)
     enumerator :: TEXTURE_FILTER_POINT = 0               ! No filter just pixel approximation
     enumerator :: TEXTURE_FILTER_BILINEAR                ! Linear filtering
     enumerator :: TEXTURE_FILTER_TRILINEAR               ! Trilinear filtering (linear with mipmaps)
     enumerator :: TEXTURE_FILTER_ANISOTROPIC_4X          ! Anisotropic filtering 4x
     enumerator :: TEXTURE_FILTER_ANISOTROPIC_8X          ! Anisotropic filtering 8x
     enumerator :: TEXTURE_FILTER_ANISOTROPIC_16X         ! Anisotropic filtering 16x
  end enum

  enum, bind(C)
    enumerator :: SHADER_UNIFORM_FLOAT = 0       ! Shader uniform type: float
    enumerator :: SHADER_UNIFORM_VEC2            ! Shader uniform type: vec2 (2 float)
    enumerator :: SHADER_UNIFORM_VEC3            ! Shader uniform type: vec3 (3 float)
    enumerator :: SHADER_UNIFORM_VEC4            ! Shader uniform type: vec4 (4 float)
    enumerator :: SHADER_UNIFORM_INT             ! Shader uniform type: int
    enumerator :: SHADER_UNIFORM_IVEC2           ! Shader uniform type: ivec2 (2 int)
    enumerator :: SHADER_UNIFORM_IVEC3           ! Shader uniform type: ivec3 (3 int)
    enumerator :: SHADER_UNIFORM_IVEC4           ! Shader uniform type: ivec4 (4 int)
    enumerator :: SHADER_UNIFORM_SAMPLER2D       ! Shader uniform type: sampler2d
 end enum

  interface
    !  LoadTextureFromImage(Image image)
         function load_texture_from_image(eximg) bind(C, name="LoadTextureFromImage")
         import :: Image, Texture
         type(Image) :: eximg
         type(Texture) :: load_texture_from_image
         end function load_texture_from_image
    ! Image LoadImage(const char *fileName)
        function load_image(file_name) bind(C, name="LoadImage")
        use iso_c_binding, only: c_char
        import :: Image
        character(kind=c_char) :: file_name(*)
        type(Image) :: load_image
        end function load_image
        subroutine unload_image(img) bind(C, name="UnloadImage")
            import :: Image
            type(Image) :: img
        end subroutine unload_image
        subroutine unload_texture(tex) bind(C, name="UnloadTexture")
            import :: Texture
            type(Texture) :: tex
        end subroutine unload_texture
        ! RLAPI void UpdateTexture(Texture2D texture, const void *pixels);
        SUBROUTINE update_texture(txt, pixels) BIND(C, NAME="UpdateTexture")
            USE ISO_C_BINDING, ONLY: C_PTR
            IMPORT :: Texture
            TYPE(Texture) :: txt
            TYPE(C_PTR), VALUE :: pixels
        END SUBROUTINE update_texture
        ! RLAPI Image GenImageColor(int width, int height, Color color);
        function gen_image_color(width, height, color) bind(C, name="GenImageColor")
            use iso_c_binding, only: c_int32_t
            import :: Image
            type(Image) :: gen_image_color
            integer(c_int32_t), value :: width
            integer(c_int32_t), value :: height
            integer(c_int32_t),value :: color
            END function gen_image_color
     subroutine init_window(width,height,title) bind(C, name="InitWindow")
       use iso_c_binding, only: c_char, c_int
       integer(c_int),value :: width
       integer(c_int),value :: height
       character(kind=c_char) :: title(*)
     end subroutine init_window

     subroutine set_target_fps(fps) bind(C, name="SetTargetFPS")
       use iso_c_binding, only: c_int
       integer(c_int),value :: fps
     end subroutine set_target_fps
     ! RLAPI void ImageDrawCircle(Image *dst, int centerX, int centerY, int radius, Color color);               // Draw a filled circle within an image
     SUBROUTINE image_draw_circle(dst, centerX, centerY, radius, color) bind(C, name="ImageDrawCircle")
         use iso_c_binding, only: c_int32_t
         import :: Image
         integer(c_int32_t),value :: color
         type(image) :: dst
         integer(c_int32_t) ,value :: centerX, centerY, radius
     END SUBROUTINE image_draw_circle
     logical(c_bool) function window_should_close() bind(C, name="WindowShouldClose")
       use iso_c_binding, only: c_bool
     end function window_should_close

     real(c_float) function get_frame_time() bind(C, name="GetFrameTime")
       use iso_c_binding, only: c_float
     end function get_frame_time

     subroutine begin_drawing() bind(C, name="BeginDrawing")
     end subroutine begin_drawing

     subroutine end_drawing() bind(C, name="EndDrawing")
     end subroutine end_drawing

     subroutine clear_background(color) bind(C, name="ClearBackground")
       use iso_c_binding, only: c_int32_t
       integer(c_int32_t),value :: color
     end subroutine clear_background

     subroutine draw_rectangle(x,y,w,h,color) bind(C, name="DrawRectangle")
       use iso_c_binding, only: c_int, c_int32_t
       integer(c_int),value :: x
       integer(c_int),value :: y
       integer(c_int),value :: w
       integer(c_int),value :: h
       integer(c_int32_t),value :: color
     end subroutine draw_rectangle

     subroutine set_config_flags(flags) bind(C, name="SetConfigFlags")
       use iso_c_binding, only: c_int32_t
       integer(c_int32_t),value :: flags
     end subroutine set_config_flags

     integer(c_int) function get_render_width() bind(C, name="GetRenderWidth")
       use iso_c_binding, only: c_int
     end function get_render_width

     integer(c_int) function get_render_height() bind(C, name="GetRenderHeight")
       use iso_c_binding, only: c_int
     end function get_render_height

     integer(c_int) function get_mouse_x() bind(C, name="GetMouseX")
       use iso_c_binding, only: c_int
     end function get_mouse_x

     integer(c_int) function get_mouse_y() bind(C, name="GetMouseY")
       use iso_c_binding, only: c_int
     end function get_mouse_y

     subroutine draw_line(startPosX, startPosY, endPosX, endPosY, color) bind(C, name="DrawLine")
         use iso_c_binding, only: c_int, c_int32_t
         integer(c_int), value :: startPosX
         integer(c_int), value :: startPosY
         integer(c_int), value :: endPosX
         integer(c_int), value :: endPosY
         integer(c_int32_t),value :: color
     END subroutine draw_line

     subroutine draw_line_ex(startPos,endPos,thick,color) bind(C, name="DrawLineEx")
       use iso_c_binding, only: c_int, c_float, c_int32_t
       import :: Vector2
       type(Vector2),value      :: startPos
       type(Vector2),value      :: endPos
       real(c_float),value      :: thick
       integer(c_int32_t),value :: color
     end subroutine draw_line_ex

     ! RLAPI void DrawRing(Vector2 center, float innerRadius, float outerRadius, float startAngle, float endAngle, int segments, Color color); // Draw ring
     subroutine draw_ring(center,innerRadius,outerRadius,startAngle,endAngle,segments,color) bind(C, name="DrawRing")
       use iso_c_binding, only: c_float, c_int, c_int32_t
       import :: Vector2
       type(Vector2),value      :: center
       real(c_float),value      :: innerRadius
       real(c_float),value      :: outerRadius
       real(c_float),value      :: startAngle
       real(c_float),value      :: endAngle
       integer(c_int),value     :: segments
       integer(c_int32_t),value :: color
     end subroutine draw_ring

     ! RLAPI void DrawCircleV(Vector2 center, float radius, Color color);                                       // Draw a color-filled circle (Vector version)
     subroutine draw_circle_v(center, radius, color) bind(C, name="DrawCircleV")
       use iso_c_binding, only: c_float, c_int32_t
       import :: Vector2
       type(Vector2), value     :: center
       real(c_float), value     :: radius
       integer(c_int32_t), value:: color
     end subroutine draw_circle_v

     subroutine draw_circle(centerX, centerY, radius, color) bind(C, name="DrawCircle")
       use iso_c_binding, only: c_float, c_int32_t
       real(c_float), value     :: radius
       integer(c_int32_t), value:: color, centerX, centerY
     end subroutine draw_circle

     logical(c_bool) function is_key_pressed(key) bind(C, name="IsKeyPressed")
       use iso_c_binding, only: c_int, c_bool
       integer(c_int),value :: key
     end function is_key_pressed


     logical(c_bool) function is_mouse_button_pressed(button) bind(C, name="IsMouseButtonPressed")
       use iso_c_binding, only: c_int, c_bool
       integer(c_int),value :: button
     end function is_mouse_button_pressed

     logical(c_bool) function is_mouse_button_down(button) bind(C, name="IsMouseButtonDown")
       use iso_c_binding, only: c_int, c_bool
       integer(c_int),value :: button
     end function is_mouse_button_down

     logical(c_bool) function is_mouse_button_released(button) bind(C, name="IsMouseButtonReleased")
       use iso_c_binding, only: c_int, c_bool
       integer(c_int),value :: button
     end function is_mouse_button_released

     ! RLAPI void DrawText(const char *text, int posX, int posY, int fontSize, Color color);       // Draw text (using default font)
     subroutine draw_text(text,posX,posY,fontSize,color) bind(C, name="DrawText")
       use iso_c_binding, only: c_char, c_int, c_int32_t
       character(kind=c_char) :: text(*)
       integer(c_int),value :: posX, posY, fontSize
       integer(c_int32_t),value :: color
     end subroutine draw_text

     ! RLAPI Font LoadFont(const char *fileName);                                                  // Load font from file into GPU memory (VRAM)
     type(Font) function load_font(fileName) bind(C, name="LoadFont")
       use iso_c_binding, only: c_char
       import :: Font
       character(kind=c_char) :: fileName(*)
     end function load_font

     ! RLAPI void DrawTextEx(Font font, const char *text, Vector2 position, float fontSize, float spacing, Color tint); // Draw text using font and additional parameters
     subroutine draw_text_ex(text_font, text, position, fontSize, spacing, tint) bind(C, name="DrawTextEx")
       use iso_c_binding, only: c_char, c_float, c_int32_t
       import :: Font, Vector2
       type(Font),value         :: text_font
       character(kind=c_char)   :: text(*)
       type(Vector2),value      :: position
       real(c_float),value      :: fontSize, spacing
       integer(c_int32_t),value :: tint
     end subroutine draw_text_ex

     !RLAPI Font LoadFontEx(const char *fileName, int fontSize, int *fontChars, int glyphCount);  // Load font from file with extended parameters, use NULL for fontChars and 0 for glyphCount to load the default character set
     type(Font) function load_font_ex(fileName, fontSize, fontChars, glyphCount) bind(C, name="LoadFontEx")
       use iso_c_binding, only: c_char, c_int, c_ptr
       import :: Font
       character(kind=c_char) :: fileName
       integer(c_int),value   :: fontSize
       type(c_ptr),value      :: fontChars
       integer(c_int),value   :: glyphCount
     end function load_font_ex

     ! RLAPI void DrawRectangleRounded(Rectangle rec, float roundness, int segments, Color color);              // Draw rectangle with rounded edges
     subroutine draw_rectangle_rounded(rec, roundness, segments, color) bind(C, name="DrawRectangleRounded")
       use iso_c_binding, only: c_float, c_int, c_int32_t
       import :: Rectangle
       type(Rectangle),value :: rec
       real(c_float),value :: roundness
       integer(c_int),value :: segments
       integer(c_int32_t),value :: color
     end subroutine draw_rectangle_rounded

     ! RLAPI Vector2 GetMousePosition(void);                         // Get mouse position XY
     type(Vector2) function get_mouse_position() bind(C, name="GetMousePosition")
       import :: Vector2
     end function get_mouse_position

     ! RLAPI bool CheckCollisionPointRec(Vector2 point, Rectangle rec);                                         // Check if point is inside rectangle
     logical(c_bool) function check_collision_point_rect(point, rec) bind(C, name="CheckCollisionPointRec")
       use iso_c_binding, only: c_bool
       import :: Vector2, Rectangle
       type(Vector2),value :: point
       type(Rectangle),value :: rec
     end function check_collision_point_rect

     ! RLAPI Vector2 MeasureTextEx(Font font, const char *text, float fontSize, float spacing);    // Measure string size for Font
     type(Vector2) function measure_text_ex(text_font, text, fontSize, spacing) bind(C, name="MeasureTextEx")
       use iso_c_binding, only: c_char, c_float
       import :: Vector2, Font
       type(Font),value       :: text_font
       character(kind=c_char) :: text(*)
       real(c_float),value    :: fontSize, spacing
     end function measure_text_ex

     integer(c_int32_t) function measure_text(text, fontSize) bind(C, name="MeasureText")
       use iso_c_binding, only: c_char, c_int32_t
       character(kind=c_char) :: text(*)
       integer(c_int32_t), value :: fontSize
     end function measure_text

     ! RLAPI Color ColorBrightness(Color color, float factor);                     // Get color with brightness correction, brightness factor goes from -1.0f to 1.0f
     integer(c_int32_t) function color_brightness(color, factor) bind(C, name="ColorBrightness")
       use iso_c_binding, only: c_float, c_int32_t
       integer(c_int32_t), value :: color
       real(c_float), value :: factor
     end function color_brightness

     !RLAPI Color ColorAlpha(Color color, float alpha);                           // Get color with alpha applied, alpha goes from 0.0f to 1.0f
     integer(c_int32_t) function color_alpha(color, alpha) bind(C, name="ColorAlpha")
       use iso_c_binding, only: c_float, c_int32_t
       integer(c_int32_t), value :: color
       real(c_float), value :: alpha
     end function color_alpha

     ! RLAPI void SetTextureFilter(Texture2D texture, int filter);                                              // Set texture scaling filter mode
     subroutine set_texture_filter(input_texture, filter) bind(C, name="SetTextureFilter")
       use iso_c_binding, only: c_int
       import :: Texture
       type(Texture), value :: input_texture
       integer(c_int), value :: filter
     end subroutine set_texture_filter

     ! RLAPI void DrawRectangleRec(Rectangle rec, Color color);                                                 // Draw a color-filled rectangle
     subroutine draw_rectangle_rec(rec, color) bind(C, name="DrawRectangleRec")
       use iso_c_binding, only: c_int32_t
       import :: Rectangle
       type(Rectangle), value :: rec
       integer(c_int32_t), value :: color
     end subroutine draw_rectangle_rec

     ! RLAPI void DrawRectangleLinesEx(Rectangle rec, float lineThick, Color color);                            // Draw rectangle outline with extended parameters
     subroutine draw_rectangle_lines_ex(rec, lineThick, color) bind(C, name="DrawRectangleLinesEx")
       use iso_c_binding, only: c_float, c_int32_t
       import :: Rectangle
       type(Rectangle), value :: rec
       real(c_float), value :: lineThick
       integer(c_int32_t), value :: color
     end subroutine draw_rectangle_lines_ex

     ! RLAPI RenderTexture LoadRenderTexture(int width, int height);                                          // Load texture for rendering (framebuffer)
     type(RenderTexture) function load_render_texture(width,height) bind(C, name="LoadRenderTexture")
       use iso_c_binding, only: c_int
       import :: RenderTexture
       integer(c_int),value :: width, height
     end function load_render_texture

     ! RLAPI void BeginTextureMode(RenderTexture2D target);              // Begin drawing to render texture
     subroutine begin_texture_mode(target) bind(C, name="BeginTextureMode")
       import :: RenderTexture
       type(RenderTexture),value :: target
     end subroutine begin_texture_mode

     ! RLAPI void EndTextureMode(void);                                  // Ends drawing to render texture
     subroutine end_texture_mode() bind(C, name="EndTextureMode")
     end subroutine end_texture_mode

     ! RLAPI void DrawTexture(Texture2D texture, int posX, int posY, Color tint);                               // Draw a Texture2D
     subroutine draw_texture(tex,posX,posY,tint) bind(C, name="DrawTexture")
       use iso_c_binding, only: c_int, c_int32_t
       import :: Texture
       type(Texture),value :: tex
       integer(c_int),value :: posX, posY
       integer(c_int32_t),value :: tint
     end subroutine draw_texture

     ! RLAPI void DrawTextureEx(Texture2D texture, Vector2 position, float rotation, float scale, Color tint);  // Draw a Texture2D with extended parameters
     subroutine draw_texture_ex(tex, position, rotation, scale, tint) bind(C, name="DrawTextureEx")
       use iso_c_binding, only: c_float, c_int32_t
       import :: Texture, Vector2
       type(Texture), value :: tex
       type(Vector2), value :: position
       real(c_float), value :: rotation, scale
       integer(c_int32_t), value :: tint
     end subroutine draw_texture_ex

     !RLAPI Shader LoadShader(const char *vsFileName, const char *fsFileName);   // Load shader from files and bind default locations
     type(Shader) function load_shader(vsFileName, fsFileName) bind(C, name="LoadShader")
       use iso_c_binding, only: c_char
       import :: Shader
       character(kind=c_char) :: vsFileName(*), fsFileName(*)
     end function load_shader

     ! RLAPI void BeginShaderMode(Shader shader);                        // Begin custom shader drawing
     subroutine begin_shader_mode(shad) bind(C, name="BeginShaderMode")
       import :: Shader
       type(Shader), value :: shad
     end subroutine begin_shader_mode

     ! RLAPI void EndShaderMode(void);                                   // End custom shader drawing (use default shader)
     subroutine end_shader_mode() bind(C, name="EndShaderMode")
     end subroutine end_shader_mode

     ! RLAPI void SetMouseOffset(int offsetX, int offsetY);          // Set mouse offset
     subroutine set_mouse_offset(offsetX, offsetY) bind(C, name="SetMouseOffset")
       use iso_c_binding, only: c_int
       integer(c_int),value :: offsetX, offsetY
     end subroutine set_mouse_offset

     ! RLAPI void SetMouseScale(float scaleX, float scaleY);         // Set mouse scaling
     subroutine set_mouse_scale(scaleX, scaleY) bind(C, name="SetMouseScale")
       use iso_c_binding, only: c_float
       real(c_float),value :: scaleX, scaleY
     end subroutine set_mouse_scale

     ! RLAPI int GetShaderLocation(Shader shader, const char *uniformName);       // Get shader uniform location
     integer(c_int) function get_shader_location(shad, uniformName) bind(C, name="GetShaderLocation")
       use iso_c_binding, only: c_char, c_int
       import :: Shader
       type(Shader), value :: shad
       character(kind=c_char) :: uniformName(*)
     end function get_shader_location

     ! RLAPI void SetShaderValue(Shader shader, int locIndex, const void *value, int uniformType);               // Set shader uniform value
     subroutine set_shader_value(shad, locIndex, val, uniformType) bind(C, name="SetShaderValue")
       use iso_c_binding, only: c_int, c_ptr
       import :: Shader
       type(Shader), value :: shad
       integer(c_int), value :: locIndex
       type(c_ptr), value :: val
       integer(c_int), value :: uniformType
     end subroutine set_shader_value

     ! RLAPI void BeginMode2D(Camera2D camera);                          // Begin 2D mode with custom camera (2D)
     subroutine begin_mode_2d(camera) bind(C, name="BeginMode2D")
       import :: Camera2D
       type(Camera2D),value :: camera
     end subroutine begin_mode_2d

     ! RLAPI void EndMode2D(void);                                       // Ends 2D mode with custom camera
     subroutine end_mode_2d() bind(C, name="EndMode2D")
     end subroutine end_mode_2d

     ! RLAPI Sound LoadSound(const char *fileName);                          // Load sound from file
     type(Sound) function load_sound(fileName) bind(C, name="LoadSound")
       use iso_c_binding, only: c_char
       import :: Sound
       character(kind=c_char) :: fileName
     end function load_sound
     ! RLAPI void InitAudioDevice(void);                                     // Initialize audio device and context
     subroutine init_audio_device() bind(C, name="InitAudioDevice")
     end subroutine init_audio_device

     ! RLAPI void PlaySound(Sound sound);                                    // Play a sound
     subroutine play_sound(snd) bind(C, name="PlaySound")
       import :: Sound
       type(Sound),value :: snd
     end subroutine play_sound
  end interface
end module raylib

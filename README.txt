CHARACTER GIF MAKER
===================
Pick an anime video, describe a character, and get a smooth montage GIF of only that character.

HOW TO USE
1. Double-click "Character GIF Maker.bat".
2. Browse: choose an episode (or several episodes at once).
3. Type who you want, separated by commas, for example:
       brown hair, red eyes, girl, wearing a white dress
       dark green hair, girl
       black hair, stubble, man
4. Press Make GIF. The first time a video is used it reads the whole video (about 2-4 minutes for an
   episode). After that the same video opens straight away.

Press "Find her" first if you want to see who it found before making anything. The best match is
highlighted. If it picked the wrong person, click the right one in the list and press Make GIF.
Your click always beats the description.

Every press of Make GIF chooses different moments, so press it again for another version.

THE 10 STYLES
Slide & settle   Slams in from the side, blurry and bright, then settles sharp
Zoom punch       Crashes in or out at the face, then a slow push
Whip pan         Fast smeared swipes that carry from one shot into the next
Dreamy drift     Slow floating moves, soft dissolves and a gentle glow
Flash cuts       Quick, hard cuts, each with a white flash and a small pop
Spin settle      Twists into place with a zoom and keeps a small tilt
Impact shake     Hits like a beat drop: camera shake and colour split
Elastic bounce   Springs in and wobbles to a stop, playful
Step zoom        Three punch-ins on the same shot, closer each time
Push slide       Each shot pushes the last one off screen, smooth and clean

Mix              A different move on every shot
All 10 styles    Makes one GIF in each style, with different moments in each

SETTINGS
Length (s)      How long the GIF runs. Each style fits as many shots as it needs into this.
Pace            Fast, Normal or Slow shots.
Shape           Portrait, Square, Phone (tall) or Wide.
Smoothness      Frames per second. 25 is smooth; 15 gives smaller files; 33 and 50 are very smooth but big.
Height          Picture height in pixels before any size fitting.
Max size (MB)   The GIF is made to fit under this. It first drops dithering (hard to notice on anime),
                then shrinks the picture. The result list shows the final size.

HOW IT KEEPS TO THE SAME CHARACTER
It finds every anime face in the video and groups the faces by character with a model trained to
recognise anime characters. Your description only chooses WHICH group. Every moment is then checked
against that character again before it goes in, and doubtful ones are left out.

The hair and eye words are the strongest clues. Clothes words work less reliably: it prefers scenes
that seem to match, but check the picks. Eye colour is the least reliable. If the description
keeps choosing someone else, click the right person in the list.

GOOD TO KNOW
- Made for anime / 2D faces. It will not find 3D or live-action people.
- Characters who are on screen only briefly may give fewer moments, and then the GIF comes out shorter.
- Very similar-looking characters can get grouped together. The list shows sample faces so you can see.
- Opening and ending songs count as part of the video, so a shot can show credits text over her.
  Press Make GIF again for other moments, or use a video without them.
- GIFs are saved in the "output" folder unless you choose another. "Open folder" opens it.
- Your videos are only read, never changed.

SETUP
Windows, Python 3.11 in its default install folder, and ffmpeg on the PATH.
    pip install -r requirements.txt
The models are read from the Hugging Face cache and are not downloaded by the app. Get them once with:
    hf download deepghs/anime_face_detection
    hf download deepghs/ccip_onnx
    hf download openai/clip-vit-base-patch32

The .bat uses the Python at Desktop\TagGUI Plus\venv if that folder exists, otherwise the normal Python.
Without torch and transformers the app still runs, but the description uses only colours. Without a
GPU build of onnxruntime it runs on the CPU, which is slower.

This repository provides the basic software framework to explore QP-based motion generation in my advanced robotics lecture.

Run as follows:

```bash
ros2 launch demo.launch.xml
./qt_controller.py mode ...
```

supporting the following modes:
- `plane`
	- `dist`: keep distance to center
	- `rotate`: rotate about plane normal
	- `rotateCenter`: rotate about plane center
- `box` mode: constrain position to box
- `parallel`: eef's y-axes aligned with world's z
- `cone`: as parallel, but with slack

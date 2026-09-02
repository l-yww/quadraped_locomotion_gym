import sys
import mujoco

urdf_path = sys.argv[1] if len(sys.argv) > 1 else "legged_gym/resources/robots/cowa2/urdf/cowa2.urdf"
xml_path = sys.argv[2] if len(sys.argv) > 2 else "sim2sim/cowa2_description_mujoco/xml/cowa2.xml"

model = mujoco.MjModel.from_xml_path(urdf_path)
mujoco.mj_saveLastXML(xml_path, model)
print(f"Saved XML to {xml_path}")

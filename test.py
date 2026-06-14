import carla

client = carla.Client("192.168.56.1", 2000)
client.set_timeout(5.0)

try:
    world = client.get_world()
    print("Connected!")
    print("Map:", world.get_map().name)

except Exception as e:
    print("Failed:", e)
using UnityEngine;
using System.Collections.Generic;

[System.Serializable]
public class PlaneTransform
{
    public Vector3 position;
    public Quaternion rotation;
    public Vector3 scale;
}

[System.Serializable]
public class PlaneTransformList{
    public PlaneTransform[] list;

}

public class GeneratePlanes : MonoBehaviour
{
    string filepath = "Assets/plane_transforms.json";

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        // Read in json
        string jsonString = System.IO.File.ReadAllText(filepath);
        jsonString = "{\"list\":" + jsonString + "}";
        PlaneTransformList planeTransformList = JsonUtility.FromJson<PlaneTransformList>(jsonString);

        Debug.Log("Number of planes: " + planeTransformList.list.Length);


        foreach (var planeTransform in planeTransformList.list)
        {
            GameObject plane = GameObject.CreatePrimitive(PrimitiveType.Plane);
            plane.transform.position = new Vector3(planeTransform.position.x, planeTransform.position.z, planeTransform.position.y);
            plane.transform.rotation = planeTransform.rotation;
            plane.transform.localScale = new Vector3(planeTransform.scale.x, planeTransform.scale.z, planeTransform.scale.y);
        }

        Debug.Log("all planes loaded successfully");
    }

}

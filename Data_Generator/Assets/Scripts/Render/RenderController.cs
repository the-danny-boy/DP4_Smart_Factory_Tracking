using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RenderController : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        Physics.gravity = new Vector3(0f, 0f, 0f);
        Random.InitState(42);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}

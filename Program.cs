using FaceONNX;
using System.Drawing;
using System.IO;
using UMapx.Core;
using UMapx.Imaging;
// using UMapx.Visualization;

using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Newtonsoft.Json;

namespace GenderClassification
{
    class Program
    {
        static void Main()
        {
            // create a new web host instance
            var host = new WebHostBuilder()
                .UseKestrel() // use Kestrel web server
                .UseUrls("http://localhost:5000") // set the server URL
                .Configure(app =>
                {
                    // handle incoming HTTP requests
                    app.Run(async context =>
                    {
                        try{
                            // get the request body as bytes
                            byte[] imageData = new byte[context.Request.ContentLength ?? 0];
                            await context.Request.Body.ReadAsync(imageData);
                            using (var image = Image.FromStream(new MemoryStream(imageData)))
                            {
                                var jsonResponse = GetPeoplesAttributesFromImage(image);
                                context.Response.ContentType = "application/json";
                                await context.Response.WriteAsync(jsonResponse);
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Error: {ex}");
                        }
                    });
                })
                .Build();

            // start the web host
            host.Run();
        }

        private static string GetPeoplesAttributesFromImage(Image image)
        {
            using var faceDetector = new FaceDetector();
            using var faceGenderClassifier = new FaceGenderClassifier();
            var labels = FaceGenderClassifier.Labels;
            // using var painter = new Painter() {
            //     BoxPen = new Pen(Color.Yellow, 4),
            //     Transparency = 0,
            // };

            using var bitmap = new Bitmap(image);
            var faces = faceDetector.Forward(bitmap);
            // int i = 1;

            // JSONオブジェクトのリストを定義
            List<object> jsonList = new List<object>();
            
            foreach (var face in faces)
            {
                // Console.Write($"\t[Face #{i++}]: ");

                // var paintData = new PaintData()
                // {
                //     Rectangle = face,
                //     Title = string.Empty
                // };
                // var graphics = Graphics.FromImage(bitmap);
                // painter.Draw(graphics, paintData);

                var cropped = BitmapTransform.Crop(bitmap, face);
                var output = faceGenderClassifier.Forward(cropped);
                var max = Matrice.Max(output, out int gender);
                var label = labels[gender];

                // JSONオブジェクトを作成してリストに追加
                var jsonObject = new { label = label, score = output.Max() };
                jsonList.Add(jsonObject);

                // Console.WriteLine($"--> classified as [{label}] gender with probability [{output.Max()}]");
            }

            // bitmap.Save(Path.Combine("output", "result.png"));

            // JSON文字列にシリアライズ
            string jsonString = JsonConvert.SerializeObject(jsonList);

            return jsonString;
        }
    }
}
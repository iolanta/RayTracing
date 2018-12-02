using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace RayTracing
{
    public partial class Form1 : Form
    {
        public List<Figure> scene = new List<Figure>();
        public List<Light> lights = new List<Light>();   // список источников света
        public Color[,] color_pixels;                    // цвета пикселей для отображения на pictureBox
        public Point3D[,] pixels;
        public Point3D focus;
        public Point3D up_left, up_right, down_left, down_right;
        public int h, w;

        public Form1()
        {
            InitializeComponent();
            focus = new Point3D();
            up_left = new Point3D();
            up_right = new Point3D();
            down_left = new Point3D();
            down_right = new Point3D();
            h = pictureBox1.Height;
            w = pictureBox1.Width;
            pictureBox1.Image = new Bitmap(w, h);
        }

        public void build_scene()
        {
            Figure room = Figure.get_Hexahedron(10);
            up_left = room.sides[0].get_point(0);
            up_right = room.sides[0].get_point(1);
            down_right = room.sides[0].get_point(2);
            down_left = room.sides[0].get_point(3);

            Point3D normal = Side.norm(room.sides[0]);                            // нормаль стороны комнаты
            Point3D center = (up_left + up_right + down_left + down_right) / 4;   // центр стороны комнаты
            focus = center - normal * 10;

            room.set_pen(new Pen(Color.Gray));
            room.sides[3].drawing_pen = new Pen(Color.Fuchsia);
            room.sides[2].drawing_pen = new Pen(Color.Blue);
            room.sides[1].drawing_pen = new Pen(Color.Yellow);
            room.figure_material = new Material(0, 0, 0.6f, 2f);

            Light l = new Light(new Point3D(0, 0, 4f), new Point3D(1f, 1f, 1f));

            Sphere s = new Sphere(new Point3D(0, 0, 0), 1f);
            s.set_pen(new Pen(Color.Green));
            s.figure_material = new Material(0, 0.5f, 0f, 0f, 2f);
            s.offset(0.5f, -2, -2);

            Figure cube = Figure.get_Hexahedron(2);
            cube.offset(0, 1, -3.5f);
            cube.set_pen(new Pen(Color.Red));
            cube.figure_material = new Material(0f, 1f, 0.4f, 0.2f);
            
            scene.Add(room);
            scene.Add(cube);
            scene.Add(s);
            lights.Add(l);
        }

        private void button1_Click(object sender, EventArgs e)
        {
            build_scene();
            run_rayTrace();

            for (int i = 0; i < w; ++i)
            {
                for (int j = 0; j < h; ++j)
                {
                    (pictureBox1.Image as Bitmap).SetPixel(i, j, color_pixels[i, j]);
                }
                pictureBox1.Invalidate();
            }

        }

        public void run_rayTrace()
        {
            get_pixels();
            for(int i = 0; i < w; ++i)
 
                for(int j = 0; j < h; ++j)
                {
                    Ray r = new Ray(focus, pixels[i, j]);
                    r.start = new Point3D(pixels[i, j]);
                    Point3D clr = RayTrace(r, 10, 1);
                    if (clr.x > 1.0f || clr.y > 1.0f || clr.z > 1.0f)
                        clr = Point3D.norm(clr);
                    color_pixels[i, j] = Color.FromArgb((int)(255 * clr.x), (int)(255 * clr.y), (int)(255 * clr.z));
                }
        }

        // получение всех пикселей сцены
        public void get_pixels()
        {
            pixels = new Point3D[w, h];
            color_pixels = new Color[w, h];
            Point3D step_up = (up_right - up_left) / (w - 1);
            Point3D step_down = (down_right - down_left) / (w - 1);

            Point3D up = new Point3D(up_left);
            Point3D down = new Point3D(down_left);

            for (int i = 0; i < w; ++i)
            {
                Point3D step_y = (up - down) / (h - 1);
                Point3D d = new Point3D(down);
                for (int j = 0; j < h; ++j)
                {
                    pixels[i, j] = d;
                    d += step_y;
                }
                up += step_up;
                down += step_down;
            }
        }

        // видима ли точка пересечения луча с фигурой из источника света
        public bool is_visible(Point3D light_point, Point3D hit_point)
        {
            float max_t = (light_point - hit_point).length();     // позиция источника света на луче
            Ray r = new Ray(hit_point, light_point);

            foreach(Figure fig in scene)
                if (fig.figure_intersection(r, out float t, out Point3D n))
                    if (t < max_t && t > Figure.EPS)
                        return false;
             return true;
        }

        public Point3D RayTrace(Ray r, int iter, float env)
        {
            if (iter <= 0)
                return new Point3D(0, 0, 0);

            float t = 0;     // позиция точки пересечения луча с фигурой на луче
            Point3D normal = null;
            Material m = new Material();
            Point3D res_color = new Point3D(0, 0, 0);
            bool refract_out_of_figure = false;

            foreach(Figure fig in scene)
            {
                if (fig.figure_intersection(r, out float intersect, out Point3D n))
                    if(intersect < t || t == 0)   // нужна ближайшая фигура к точке наблюдения
                    {
                        t = intersect;
                        normal = n;
                        m = new Material(fig.figure_material);
                    }
            }

            if (t == 0)
                return new Point3D(0, 0, 0);

            if (Point3D.scalar(r.direction, normal) > 0)
            {
                normal *= -1; 
                refract_out_of_figure = true;
            }

            Point3D hit_point = r.start + r.direction * t;

            foreach(Light l in lights)
            {
                Point3D amb = l.color_light * m.ambient;
                amb.x = (amb.x * m.clr.x);
                amb.y = (amb.y * m.clr.y);
                amb.z = (amb.z * m.clr.z);
                res_color += amb;

                if (is_visible(l.point_light, hit_point))
                    res_color += l.shade(hit_point, normal, m.clr, m.diffuse);
            }

            if(m.reflection > 0)
            {
                Ray reflected_ray = r.reflect(hit_point, normal);
                res_color += m.reflection * RayTrace(reflected_ray, iter - 1, env);
            }

            if(m.refraction > 0)
            {
                float eta;
                if (refract_out_of_figure)
                    eta = m.environment;
                else
                    eta = 1 / m.environment;

                Ray refracted_ray = r.refract(hit_point, normal, eta);
                if(refracted_ray != null)
                    res_color += m.refraction * RayTrace(refracted_ray, iter - 1, m.environment);
            }

            return res_color;
        }
    }

    public class Ray
    {
        public Point3D start, direction;

        public Ray(Point3D st, Point3D end)
        {
            start = new Point3D(st);
            direction = Point3D.norm(end - st);
        }

        public Ray() { }

        public Ray(Ray r)
        {
            start = r.start;
            direction = r.direction;
        }

        public Ray reflect(Point3D hit_point, Point3D normal)
        {
            Point3D reflect_dir = direction - 2 * normal * Point3D.scalar(direction, normal);
            return new Ray(hit_point, hit_point + reflect_dir);
        }

        public Ray refract(Point3D hit_point, Point3D normal, float eta)
        {
            Ray res_ray = new Ray();
            float sclr = Point3D.scalar(normal, direction);

            float k = 1 - eta * eta * (1 - sclr * sclr);

            if (k >= 0)
            {
                float cos_theta = (float)Math.Sqrt(k);
                res_ray.start = new Point3D(hit_point);
                res_ray.direction = Point3D.norm(eta * direction - (cos_theta + eta * sclr) * normal);
                return res_ray;
            }
            else
                return null;
        }
    }

    public class Material
    {
        public float reflection;    // коэффициент отражения
        public float refraction;    // коэффициент преломления
        public float environment;   // коэффициент преломления среды
        public float ambient;       // коэффициент принятия фонового освещения
        public float diffuse;       // коэффициент принятия диффузного освещения
        public Point3D clr;         // цвет материала

        public Material(float refl, float refr, float amb, float dif, float env = 1)
        {
            reflection = refl;
            refraction = refr;
            ambient = amb;
            diffuse = dif;
            environment = env;
        }

        public Material(Material m)
        {
            reflection = m.reflection;
            refraction = m.refraction;
            environment = m.environment;
            ambient = m.ambient;
            diffuse = m.diffuse;
            clr = new Point3D(m.clr);
        }

        public Material() { }
    }

    public class Point3D
    {
        public float x, y, z;

        public Point3D()
        {
            x = 0;
            y = 0;
            z = 0;
        }
        public Point3D(float _x, float _y, float _z)
        {
            x = _x;
            y = _y;
            z = _z;
        }

        public Point3D(Point3D p)
        {
            if (p == null)
                return;
            x = p.x;
            y = p.y;
            z = p.z;
        }

        public override string ToString()
        {
            return String.Format("X:{0:f1} Y:{1:f1} Z:{2:f1}", x, y, z);
        }

        public float length()
        {
            return (float)Math.Sqrt(x * x + y * y + z * z);
        }

        public static Point3D operator -(Point3D p1, Point3D p2)
        {
            return new Point3D(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);

        }

        public static float scalar(Point3D p1, Point3D p2)
        {
            return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
        }

        public static Point3D norm(Point3D p)
        {
            float z = (float)Math.Sqrt((float)(p.x * p.x + p.y * p.y + p.z * p.z));
            if (z == 0)
                return new Point3D(p);
            return new Point3D(p.x / z, p.y / z, p.z / z);
        }

        public static Point3D operator +(Point3D p1, Point3D p2)
        {
            return new Point3D(p1.x + p2.x, p1.y + p2.y, p1.z + p2.z);

        }

        public static Point3D operator *(Point3D p1, Point3D p2)
        {
            return new Point3D(p1.y * p2.z - p1.z * p2.y, p1.z * p2.x - p1.x * p2.z, p1.x * p2.y - p1.y * p2.x);
        }

        public static Point3D operator *(float t, Point3D p1)
        {
            return new Point3D(p1.x * t, p1.y * t, p1.z * t);
        }


        public static Point3D operator *(Point3D p1, float t)
        {
            return new Point3D(p1.x * t, p1.y * t, p1.z * t);
        }

        public static Point3D operator -(Point3D p1, float t)
        {
            return new Point3D(p1.x - t, p1.y - t, p1.z - t);
        }

        public static Point3D operator -(float t, Point3D p1)
        {
            return new Point3D(t - p1.x, t - p1.y, t - p1.z);
        }

        public static Point3D operator +(Point3D p1, float t)
        {
            return new Point3D(p1.x + t, p1.y + t, p1.z + t);
        }

        public static Point3D operator +(float t, Point3D p1)
        {
            return new Point3D(p1.x + t, p1.y + t, p1.z + t);
        }

        public static Point3D operator /(Point3D p1, float t)
        {
            return new Point3D(p1.x / t, p1.y / t, p1.z / t);
        }

        public static Point3D operator /(float t, Point3D p1)
        {
            return new Point3D(t / p1.x, t / p1.y, t / p1.z);
        }
    }

    public class Side
    {
        public Figure host = null;
        public List<int> points = new List<int>();
        public Pen drawing_pen = new Pen(Color.Black);
        public Point3D Normal;

        public Side(Figure h = null)
        {
            host = h;
        }
        public Side(Side s)
        {
            points = new List<int>(s.points);
            host = s.host;
            drawing_pen = s.drawing_pen.Clone() as Pen;
            Normal = new Point3D(s.Normal);
        }
        public Point3D get_point(int ind)
        {
            if (host != null)
                return host.points[points[ind]];
            return null;
        }

        public static Point3D norm(Side S)
        {
            if (S.points.Count() < 3)
                return new Point3D(0, 0, 0);
            Point3D U = S.get_point(1) - S.get_point(0);
            Point3D V = S.get_point(S.points.Count - 1) - S.get_point(0);
            Point3D normal = V * U;
            return Point3D.norm(normal);
        }

        public void CalculateSideNormal()
        {
            Normal = norm(this);
        }
    }

    public class Figure
    {
        public static float EPS = 0.0001f;
        public List<Point3D> points = new List<Point3D>(); // точки 
        public List<Side> sides = new List<Side>();        // стороны
        public Material figure_material;
        public Figure() {}

        // redo for new members
        public Figure(Figure f)
        {
            foreach (Point3D p in f.points)
                points.Add(new Point3D(p));

            foreach (Side s in f.sides)
            {
                sides.Add(new Side(s));
                sides.Last().host = this;
            }
        }

        public bool ray_intersects_triangle(Ray r, Point3D p0, Point3D p1, Point3D p2, out float intersect)
        {
            intersect = -1;

            Point3D edge1 = p1 - p0;
            Point3D edge2 = p2 - p0;
            Point3D h = r.direction * edge2;
            float a = Point3D.scalar(edge1, h);

            if (a > -EPS && a < EPS)
                return false;       // This ray is parallel to this triangle.

            float f = 1.0f / a;
            Point3D s = r.start - p0;
            float u = f * Point3D.scalar(s, h);

            if (u < 0 || u > 1)
                return false;

            Point3D q = s * edge1;
            float v = f * Point3D.scalar(r.direction, q);

            if (v < 0 || u + v > 1)
                return false;
            // At this stage we can compute t to find out where the intersection point is on the line.
            float t = f * Point3D.scalar(edge2, q);
            if (t > EPS)
            {
                intersect = t;
                return true;
            }
            else      // This means that there is a line intersection but not a ray intersection.
                return false;
        }

        // пересечение луча с фигурой
        public virtual bool figure_intersection(Ray r, out float intersect, out Point3D normal)
        {
            intersect = 0;
            normal = null;
            Side sd = null;

            foreach(Side s in sides)
            {
                if (s.points.Count == 3)
                {
                    if (ray_intersects_triangle(r, s.get_point(0), s.get_point(1), s.get_point(2), out float t) && (intersect == 0 || t < intersect))
                    {
                        intersect = t;
                        sd = s;
                    }
                }
                else if (s.points.Count == 4)
                {
                    if (ray_intersects_triangle(r, s.get_point(0), s.get_point(1), s.get_point(3), out float t) && (intersect == 0 || t < intersect))
                    {
                        intersect = t;
                        sd = s;
                    }
                    else if (ray_intersects_triangle(r, s.get_point(1), s.get_point(2), s.get_point(3), out t) && (intersect == 0 || t < intersect))
                    {
                        intersect = t;
                        sd = s;
                    }
                }
            }

            if (intersect != 0)
            {
                normal = Side.norm(sd);
                figure_material.clr = new Point3D(sd.drawing_pen.Color.R / 255f, sd.drawing_pen.Color.G / 255f, sd.drawing_pen.Color.B / 255f);  
                return true;
            }

            return false;
        }


        ///
        /// ----------------------------- TRANSFORMS SUPPORT METHODS --------------------------------
        ///

        public float[,] get_matrix()
        {
            var res = new float[points.Count, 4];
            for (int i = 0; i < points.Count; i++)
            {
                res[i, 0] = points[i].x;
                res[i, 1] = points[i].y;
                res[i, 2] = points[i].z;
                res[i, 3] = 1;
            }
            return res;
        }

        public void apply_matrix(float[,] matrix)
        {
            for (int i = 0; i < points.Count; i++)
            {
                points[i].x = matrix[i, 0] / matrix[i, 3];
                points[i].y = matrix[i, 1] / matrix[i, 3];
                points[i].z = matrix[i, 2] / matrix[i, 3];

            }
        }

        private Point3D get_center()
        {
            Point3D res = new Point3D(0, 0, 0);
            foreach (Point3D p in points)
            {
                res.x += p.x;
                res.y += p.y;
                res.z += p.z;

            }
            res.x /= points.Count();
            res.y /= points.Count();
            res.z /= points.Count();
            return res;
        }

        ///
        /// ----------------------------- APHINE TRANSFORMS METHODS --------------------------------
        ///

        public void rotate_around_rad(float rangle, string type)
        {
            float[,] mt = get_matrix();
            Point3D center = get_center();
            switch (type)
            {
                case "CX":
                    mt = apply_offset(mt, -center.x, -center.y, -center.z);
                    mt = apply_rotation_X(mt, rangle);
                    mt = apply_offset(mt, center.x, center.y, center.z);
                    break;
                case "CY":
                    mt = apply_offset(mt, -center.x, -center.y, -center.z);
                    mt = apply_rotation_Y(mt, rangle);
                    mt = apply_offset(mt, center.x, center.y, center.z);
                    break;
                case "CZ":
                    mt = apply_offset(mt, -center.x, -center.y, -center.z);
                    mt = apply_rotation_Z(mt, rangle);
                    mt = apply_offset(mt, center.x, center.y, center.z);
                    break;
                case "X":
                    mt = apply_rotation_X(mt, rangle);
                    break;
                case "Y":
                    mt = apply_rotation_Y(mt, rangle);
                    break;
                case "Z":
                    mt = apply_rotation_Z(mt, rangle);
                    break;
                default:
                    break;
            }
            apply_matrix(mt);
        }

        public void rotate_around(float angle, string type)
        {
            rotate_around_rad(angle * (float)Math.PI / 180, type);
        }

        public void scale_axis(float xs, float ys, float zs)
        {
            float[,] pnts = get_matrix();
            pnts = apply_scale(pnts, xs, ys, zs);
            apply_matrix(pnts);
        }

        public void offset(float xs, float ys, float zs)
        {
            apply_matrix(apply_offset(get_matrix(), xs, ys, zs));
        }

        public void set_pen(Pen dw)
        {
            foreach (Side s in sides)
                s.drawing_pen = dw;

        }

        public void scale_around_center(float xs, float ys, float zs)
        {
            float[,] pnts = get_matrix();
            Point3D p = get_center();
            pnts = apply_offset(pnts, -p.x, -p.y, -p.z);
            pnts = apply_scale(pnts, xs, ys, zs);
            pnts = apply_offset(pnts, p.x, p.y, p.z);
            apply_matrix(pnts);
        }

        public void line_rotate_rad(float rang, Point3D p1, Point3D p2)
        {

            p2 = new Point3D(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
            p2 = Point3D.norm(p2);

            float[,] mt = get_matrix();
            apply_matrix(rotate_around_line(mt, p1, p2, rang));
        }

        /// <summary>
        /// rotate figure line
        /// </summary>
        /// <param name="ang">angle in degrees</param>
        /// <param name="p1">line start</param>
        /// <param name="p2">line end</param>
        public void line_rotate(float ang, Point3D p1, Point3D p2)
        {
            ang = ang * (float)Math.PI / 180;
            line_rotate_rad(ang, p1, p2);
        }


        ///
        /// ----------------------------- STATIC BACKEND FOR TRANSFROMS --------------------------------
        ///

        private static float[,] rotate_around_line(float[,] transform_matrix, Point3D start, Point3D dir, float angle)
        {
            float cos_angle = (float)Math.Cos(angle);
            float sin_angle = (float)Math.Sin(angle);
            float val00 = dir.x * dir.x + cos_angle * (1 - dir.x * dir.x);
            float val01 = dir.x * (1 - cos_angle) * dir.y + dir.z * sin_angle;
            float val02 = dir.x * (1 - cos_angle) * dir.z - dir.y * sin_angle;
            float val10 = dir.x * (1 - cos_angle) * dir.y - dir.z * sin_angle;
            float val11 = dir.y * dir.y + cos_angle * (1 - dir.y * dir.y);
            float val12 = dir.y * (1 - cos_angle) * dir.z + dir.x * sin_angle;
            float val20 = dir.x * (1 - cos_angle) * dir.z + dir.y * sin_angle;
            float val21 = dir.y * (1 - cos_angle) * dir.z - dir.x * sin_angle;
            float val22 = dir.z * dir.z + cos_angle * (1 - dir.z * dir.z);
            float[,] rotateMatrix = new float[,] { { val00, val01, val02, 0 }, { val10, val11, val12, 0 }, { val20, val21, val22, 0 }, { 0, 0, 0, 1 } };
            return apply_offset(multiply_matrix(apply_offset(transform_matrix, -start.x, -start.y, -start.z), rotateMatrix), start.x, start.y, start.z);
        }

        private static float[,] multiply_matrix(float[,] m1, float[,] m2)
        {
            float[,] res = new float[m1.GetLength(0), m2.GetLength(1)];
            for (int i = 0; i < m1.GetLength(0); i++)
            {
                for (int j = 0; j < m2.GetLength(1); j++)
                {
                    for (int k = 0; k < m2.GetLength(0); k++)
                    {
                        res[i, j] += m1[i, k] * m2[k, j];
                    }
                }
            }
            return res;
        }

        private static float[,] apply_offset(float[,] transform_matrix, float offset_x, float offset_y, float offset_z)
        {
            float[,] translationMatrix = new float[,] { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { offset_x, offset_y, offset_z, 1 } };
            return multiply_matrix(transform_matrix, translationMatrix);
        }

        private static float[,] apply_rotation_X(float[,] transform_matrix, float angle)
        {
            float[,] rotationMatrix = new float[,] { { 1, 0, 0, 0 }, { 0, (float)Math.Cos(angle), (float)Math.Sin(angle), 0 },
                { 0, -(float)Math.Sin(angle), (float)Math.Cos(angle), 0}, { 0, 0, 0, 1} };
            return multiply_matrix(transform_matrix, rotationMatrix);
        }

        private static float[,] apply_rotation_Y(float[,] transform_matrix, float angle)
        {
            float[,] rotationMatrix = new float[,] { { (float)Math.Cos(angle), 0, -(float)Math.Sin(angle), 0 }, { 0, 1, 0, 0 },
                { (float)Math.Sin(angle), 0, (float)Math.Cos(angle), 0}, { 0, 0, 0, 1} };
            return multiply_matrix(transform_matrix, rotationMatrix);
        }

        private static float[,] apply_rotation_Z(float[,] transform_matrix, float angle)
        {
            float[,] rotationMatrix = new float[,] { { (float)Math.Cos(angle), (float)Math.Sin(angle), 0, 0 }, { -(float)Math.Sin(angle), (float)Math.Cos(angle), 0, 0 },
                { 0, 0, 1, 0 }, { 0, 0, 0, 1} };
            return multiply_matrix(transform_matrix, rotationMatrix);
        }

        private static float[,] apply_scale(float[,] transform_matrix, float scale_x, float scale_y, float scale_z)
        {
            float[,] scaleMatrix = new float[,] { { scale_x, 0, 0, 0 }, { 0, scale_y, 0, 0 }, { 0, 0, scale_z, 0 }, { 0, 0, 0, 1 } };
            return multiply_matrix(transform_matrix, scaleMatrix);
        }


        ///
        /// ------------------------STATIC READY FIGURES-----------------------------
        ///

        static public Figure get_Hexahedron(float sz)
        {
            Figure res = new Figure();
            res.points.Add(new Point3D(sz / 2, sz / 2, sz / 2)); // 0 
            res.points.Add(new Point3D(-sz / 2, sz / 2, sz / 2)); // 1
            res.points.Add(new Point3D(-sz / 2, sz / 2, -sz / 2)); // 2
            res.points.Add(new Point3D(sz / 2, sz / 2, -sz / 2)); //3

            res.points.Add(new Point3D(sz / 2, -sz / 2, sz / 2)); // 4
            res.points.Add(new Point3D(-sz / 2, -sz / 2, sz / 2)); //5
            res.points.Add(new Point3D(-sz / 2, -sz / 2, -sz / 2)); // 6
            res.points.Add(new Point3D(sz / 2, -sz / 2, -sz / 2)); // 7

            Side s = new Side(res);
            s.points.AddRange(new int[] { 3, 2, 1, 0 });
            res.sides.Add(s);

            s = new Side(res);
            s.points.AddRange(new int[] { 4, 5, 6, 7 });
            res.sides.Add(s);

            s = new Side(res);
            s.points.AddRange(new int[] { 2, 6, 5, 1 });
            res.sides.Add(s);

            s = new Side(res);
            s.points.AddRange(new int[] { 0, 4, 7, 3 });
            res.sides.Add(s);

            s = new Side(res);
            s.points.AddRange(new int[] { 1, 5, 4, 0 });
            res.sides.Add(s);

            s = new Side(res);
            s.points.AddRange(new int[] { 2, 3, 7, 6 });
            res.sides.Add(s);

            return res;
        }


        ///
        /// ---------------------------------------------------------------------------------------
        ///

    }

    public class Sphere : Figure
    {
        float radius;

        public Sphere(Point3D p, float r)
        {
            points.Add(p);
            radius = r;
        }


        public static bool ray_sphere_intersection(Ray r, Point3D sphere_pos, float sphere_rad, out float t)
        {
            Point3D k = r.start - sphere_pos;
            float b = Point3D.scalar(k, r.direction);
            float c = Point3D.scalar(k, k) - sphere_rad * sphere_rad;
            float d = b * b - c;
            t = 0;

            if (d >= 0)
            {
                float sqrtd = (float)Math.Sqrt(d);
                float t1 = -b + sqrtd;
                float t2 = -b - sqrtd;

                float min_t = Math.Min(t1, t2);
                float max_t = Math.Max(t1, t2);

                t = (min_t > EPS) ? min_t : max_t;
                return t > EPS;
            }
            return false;
        }

        public override bool figure_intersection(Ray r, out float t, out Point3D normal)
        {
            t = 0;
            normal = null;

            if (ray_sphere_intersection(r, points[0], radius, out t) && (t > EPS)) {
                normal = (r.start + r.direction * t) - points[0];
                normal = Point3D.norm(normal);
                return true;
            }
            return false;
        }
    }

    public class Light : Figure           // источник света
    {
        public Point3D point_light;       // точка, где находится источник света
        public Point3D color_light;       // цвет источника света

        public Light(Point3D p, Point3D c)
        {
            point_light = new Point3D(p);
            color_light = new Point3D(c);
        }

        // вычисление локальной модели освещения
        public Point3D shade(Point3D hit_point, Point3D normal, Point3D color_obj, float diffuse_coef)
        {
            Point3D dir =  point_light - hit_point;
            dir = Point3D.norm(dir);                // направление луча из источника света в точку удара

            Point3D diff = diffuse_coef * color_light * Math.Max(Point3D.scalar(normal, dir), 0);
            return new Point3D(diff.x * color_obj.x, diff.y * color_obj.y, diff.z * color_obj.z);
        }
    }
}

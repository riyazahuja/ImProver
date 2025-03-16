/-- A morphism of quivers. As we will later have categorical functors extend this structure,
we call it a `Prefunctor`. -/
structure Prefunctor (V : Type u₁) [Quiver.{v₁} V] (W : Type u₂) [Quiver.{v₂} W] where
  /-- The action of a (pre)functor on vertices/objects. -/
  obj : V → W
  /-- The action of a (pre)functor on edges/arrows/morphisms. -/
  map : ∀ {X Y : V}, (X ⟶ Y) → (obj X ⟶ obj Y)



lemma mk_obj {V W : Type*} [Quiver V] [Quiver W] {obj : V → W} {map} {X : V} :
    (Prefunctor.mk obj map).obj X = obj X := rfl


lemma mk_map {V W : Type*} [Quiver V] [Quiver W] {obj : V → W} {map} {X Y : V} {f : X ⟶ Y} :
    (Prefunctor.mk obj map).map f = map f := rfl


@[ext (iff := false)]
theorem ext {V : Type u} [Quiver.{v₁} V] {W : Type u₂} [Quiver.{v₂} W] {F G : Prefunctor V W}
    (h_obj : ∀ X, F.obj X = G.obj X)
    (h_map : ∀ (X Y : V) (f : X ⟶ Y),
      F.map f = Eq.recOn (h_obj Y).symm (Eq.recOn (h_obj X).symm (G.map f))) : F = G := by
  /-
    V : Type u
    inst✝¹ : Quiver V
    W : Type u₂
    inst✝ : Quiver W
    F G : Prefunctor V W
    h_obj : ∀ (X : V), Eq (F.obj X) (G.obj X)
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq (F.map f) (Eq.recOn ⋯ (Eq.recOn ⋯ …
    ⊢ Eq F G
  -/
  obtain ⟨F_obj, _⟩ := F
  /-
    case mk
    V : Type u
    inst✝¹ : Quiver V
    W : Type u₂
    inst✝ : Quiver W
    G : Prefunctor V W
    F_obj : V → W
    map✝ : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (F_obj X) (F_obj Y)
    h_obj : ∀ (X : V), Eq ({ obj := F_obj, map := map✝ }.obj X) (G.obj X)
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq ({ obj := F_obj, map := map✝ }.ma …
    ⊢ Eq { obj := F_obj, map := map✝ } G
  -/
  obtain ⟨G_obj, _⟩ := G
  obtain rfl : F_obj = G_obj := by
    ext X
    apply h_obj
  /-
    case mk.mk
    V : Type u
    inst✝¹ : Quiver V
    W : Type u₂
    inst✝ : Quiver W
    F_obj : V → W
    map✝¹ map✝ : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (F_obj X) (F_obj Y)
    h_obj : ∀ (X : V), Eq ({ obj := F_obj, map := map✝¹ }.obj X) ({ obj := F_obj,  …
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq ({ obj := F_obj, map := map✝¹ }.m …
    ⊢ Eq { obj := F_obj, map := map✝¹ } { obj := F_obj, map := map✝ }
  -/
  congr
  /-
    case mk.mk.e_map
    V : Type u
    inst✝¹ : Quiver V
    W : Type u₂
    inst✝ : Quiver W
    F_obj : V → W
    map✝¹ map✝ : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (F_obj X) (F_obj Y)
    h_obj : ∀ (X : V), Eq ({ obj := F_obj, map := map✝¹ }.obj X) ({ obj := F_obj,  …
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq ({ obj := F_obj, map := map✝¹ }.m …
    ⊢ Eq map✝¹ map✝
  -/
  funext X Y f
  /-
    case mk.mk.e_map.h.h.h
    V : Type u
    inst✝¹ : Quiver V
    W : Type u₂
    inst✝ : Quiver W
    F_obj : V → W
    map✝¹ map✝ : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (F_obj X) (F_obj Y)
    h_obj : ∀ (X : V), Eq ({ obj := F_obj, map := map✝¹ }.obj X) ({ obj := F_obj,  …
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq ({ obj := F_obj, map := map✝¹ }.m …
    X Y : V
    f : Quiver.Hom X Y
    ⊢ Eq (map✝¹ f) (map✝ f)
  -/
  simpa using h_map X Y f
  /-
    🎉 no goals
  -/


/-- This may be a more useful form of `Prefunctor.ext`. -/
theorem ext' {V W : Type u} [Quiver V] [Quiver W] {F G : Prefunctor V W}
    (h_obj : ∀ X, F.obj X = G.obj X)
    (h_map : ∀ (X Y : V) (f : X ⟶ Y),
      F.map f = Quiver.homOfEq (G.map f) (h_obj _).symm (h_obj _).symm) : F = G := by
  /-
    V W : Type u
    inst✝¹ : Quiver V
    inst✝ : Quiver W
    F G : Prefunctor V W
    h_obj : ∀ (X : V), Eq (F.obj X) (G.obj X)
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq (F.map f) (Quiver.homOfEq (G.map  …
    ⊢ Eq F G
  -/
  obtain ⟨Fobj, Fmap⟩ := F
  /-
    case mk
    V W : Type u
    inst✝¹ : Quiver V
    inst✝ : Quiver W
    G : Prefunctor V W
    Fobj : V → W
    Fmap : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (Fobj X) (Fobj Y)
    h_obj : ∀ (X : V), Eq ({ obj := Fobj, map := Fmap }.obj X) (G.obj X)
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq ({ obj := Fobj, map := Fmap }.map …
    ⊢ Eq { obj := Fobj, map := Fmap } G
  -/
  obtain ⟨Gobj, Gmap⟩ := G
  /-
    case mk.mk
    V W : Type u
    inst✝¹ : Quiver V
    inst✝ : Quiver W
    Fobj : V → W
    Fmap : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (Fobj X) (Fobj Y)
    Gobj : V → W
    Gmap : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (Gobj X) (Gobj Y)
    h_obj : ∀ (X : V), Eq ({ obj := Fobj, map := Fmap }.obj X) ({ obj := Gobj, map …
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq ({ obj := Fobj, map := Fmap }.map …
    ⊢ Eq { obj := Fobj, map := Fmap } { obj := Gobj, map := Gmap }
  -/
  obtain rfl : Fobj = Gobj := funext h_obj
  /-
    case mk.mk
    V W : Type u
    inst✝¹ : Quiver V
    inst✝ : Quiver W
    Fobj : V → W
    Fmap Gmap : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (Fobj X) (Fobj Y)
    h_obj : ∀ (X : V), Eq ({ obj := Fobj, map := Fmap }.obj X) ({ obj := Fobj, map …
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq ({ obj := Fobj, map := Fmap }.map …
    ⊢ Eq { obj := Fobj, map := Fmap } { obj := Fobj, map := Gmap }
  -/
  simp only [mk.injEq, heq_eq_eq, true_and]
  /-
    case mk.mk
    V W : Type u
    inst✝¹ : Quiver V
    inst✝ : Quiver W
    Fobj : V → W
    Fmap Gmap : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (Fobj X) (Fobj Y)
    h_obj : ∀ (X : V), Eq ({ obj := Fobj, map := Fmap }.obj X) ({ obj := Fobj, map …
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq ({ obj := Fobj, map := Fmap }.map …
    ⊢ Eq Fmap Gmap
  -/
  ext X Y f
  /-
    case mk.mk.h.h.h
    V W : Type u
    inst✝¹ : Quiver V
    inst✝ : Quiver W
    Fobj : V → W
    Fmap Gmap : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (Fobj X) (Fobj Y)
    h_obj : ∀ (X : V), Eq ({ obj := Fobj, map := Fmap }.obj X) ({ obj := Fobj, map …
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq ({ obj := Fobj, map := Fmap }.map …
    X Y : V
    f : Quiver.Hom X Y
    ⊢ Eq (Fmap f) (Gmap f)
  -/
  simpa only [Quiver.homOfEq_rfl] using h_map X Y f
  /-
    🎉 no goals
  -/


/-- The identity morphism between quivers. -/
@[simps]
def id (V : Type*) [Quiver V] : Prefunctor V V where
  obj := fun X => X
  map f := f


instance (V : Type*) [Quiver V] : Inhabited (Prefunctor V V) :=
  ⟨id V⟩


/-- Composition of morphisms between quivers. -/
@[simps]
def comp {U : Type*} [Quiver U] {V : Type*} [Quiver V] {W : Type*} [Quiver W]
    (F : Prefunctor U V) (G : Prefunctor V W) : Prefunctor U W where
  obj X := G.obj (F.obj X)
  map f := G.map (F.map f)


@[simp]
theorem comp_id {U V : Type*} [Quiver U] [Quiver V] (F : Prefunctor U V) :
    F.comp (id _) = F := rfl


@[simp]
theorem id_comp {U V : Type*} [Quiver U] [Quiver V] (F : Prefunctor U V) :
    (id _).comp F = F := rfl


@[simp]
theorem comp_assoc {U V W Z : Type*} [Quiver U] [Quiver V] [Quiver W] [Quiver Z]
    (F : Prefunctor U V) (G : Prefunctor V W) (H : Prefunctor W Z) :
    (F.comp G).comp H = F.comp (G.comp H) :=
  rfl


/-- Notation for a prefunctor between quivers. -/
infixl:50 " ⥤q " => Prefunctor


/-- Notation for composition of prefunctors. -/
infixl:60 " ⋙q " => Prefunctor.comp


/-- Notation for the identity prefunctor on a quiver. -/
notation "𝟭q" => id


theorem congr_map {U V : Type*} [Quiver U] [Quiver V] (F : U ⥤q V) {X Y : U} {f g : X ⟶ Y}
    (h : f = g) : F.map f = F.map g := by
  /-
    U : Type u_1
    V : Type u_2
    inst✝¹ : Quiver U
    inst✝ : Quiver V
    F : Prefunctor U V
    X Y : U
    f g : Quiver.Hom X Y
    h : Eq f g
    ⊢ Eq (F.map f) (F.map g)
  -/
  rw [h]
  /-
    🎉 no goals
  -/


/-- An equality of prefunctors gives an equality on objects. -/
theorem congr_obj {U V : Type*} [Quiver U] [Quiver V] {F G : U ⥤q V} (e : F = G) (X : U) :
                            /-
                              U : Type u_1
                              V : Type u_2
                              inst✝¹ : Quiver U
                              inst✝ : Quiver V
                              F G : Prefunctor U V
                              e : Eq F G
                              X : U
                              ⊢ Eq (F.obj X) (G.obj X)
                            -/
    F.obj X = G.obj X := by cases e; rfl
                                     /-
                                       🎉 no goals
                                     -/


/-- An equality of prefunctors gives an equality on homs. -/
theorem congr_hom {U V : Type*} [Quiver U] [Quiver V] {F G : U ⥤q V} (e : F = G) {X Y : U}
    (f : X ⟶ Y) : Quiver.homOfEq (F.map f) (congr_obj e X) (congr_obj e Y) = G.map f := by
  /-
    U : Type u_1
    V : Type u_2
    inst✝¹ : Quiver U
    inst✝ : Quiver V
    F G : Prefunctor U V
    e : Eq F G
    X Y : U
    f : Quiver.Hom X Y
    ⊢ Eq (Quiver.homOfEq (F.map f) ⋯ ⋯) (G.map f)
  -/
  subst e
  /-
    U : Type u_1
    V : Type u_2
    inst✝¹ : Quiver U
    inst✝ : Quiver V
    F : Prefunctor U V
    X Y : U
    f : Quiver.Hom X Y
    ⊢ Eq (Quiver.homOfEq (F.map f) ⋯ ⋯) (F.map f)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Prefunctors commute with `homOfEq`. -/
@[simp]
theorem homOfEq_map {U V : Type*} [Quiver U] [Quiver V] (F : U ⥤q V) {X Y : U} (f : X ⟶ Y)
    {X' Y' : U} (hX : X = X') (hY : Y = Y') :
    F.map (Quiver.homOfEq f hX hY) =
                                                                               /-
                                                                                 U : Type u_1
                                                                                 V : Type u_2
                                                                                 inst✝¹ : Quiver U
                                                                                 inst✝ : Quiver V
                                                                                 F : Prefunctor U V
                                                                                 X Y : U
                                                                                 f : Quiver.Hom X Y
                                                                                 X' Y' : U
                                                                                 hX : Eq X X'
                                                                                 hY : Eq Y Y'
                                                                                 ⊢ Eq (F.map (Quiver.homOfEq f hX hY)) (Quiver.homOfEq (F.map f) ⋯ ⋯)
                                                                               -/
      Quiver.homOfEq (F.map f) (congr_arg F.obj hX) (congr_arg F.obj hY) := by subst hX hY; simp
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/



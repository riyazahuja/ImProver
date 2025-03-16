/-- A reflexive quiver extends a quiver with a specified arrow `id X : X ⟶ X` for each `X` in its
type of objects. We denote these arrows by `id` since categories can be understood as an extension
of refl quivers.
-/
class ReflQuiver (obj : Type u) extends Quiver.{v} obj : Type max u v where
  /-- The identity morphism on an object. -/
  id : ∀ X : obj, Hom X X


/-- Notation for the identity morphism in a category. -/
scoped notation "𝟙rq" => ReflQuiver.id  -- type as \b1


@[simp]
theorem ReflQuiver.homOfEq_id {V : Type*} [ReflQuiver V] {X X' : V} (hX : X = X') :
                                                /-
                                                  V : Type u_1
                                                  inst✝ : CategoryTheory.ReflQuiver V
                                                  X X' : V
                                                  hX : Eq X X'
                                                  ⊢ Eq (Quiver.homOfEq (CategoryTheory.ReflQuiver.id X) hX hX) (CategoryTheory.R …
                                                -/
    Quiver.homOfEq (𝟙rq X) hX hX = 𝟙rq X' := by subst hX ; rfl
                                                           /-
                                                             🎉 no goals
                                                           -/


instance catToReflQuiver {C : Type u} [inst : Category.{v} C] : ReflQuiver.{v+1, u} C :=
  { inst with }


@[simp] theorem ReflQuiver.id_eq_id {C : Type*} [Category C] (X : C) : 𝟙rq X = 𝟙 X := rfl


/-- A morphism of reflexive quivers called a `ReflPrefunctor`. -/
structure ReflPrefunctor (V : Type u₁) [ReflQuiver.{v₁} V] (W : Type u₂) [ReflQuiver.{v₂} W]
    extends Prefunctor V W where
  /-- A functor preserves identity morphisms. -/
  map_id : ∀ X : V, map (𝟙rq X) = 𝟙rq (obj X) := by aesop_cat


lemma mk_obj {V W : Type*} [ReflQuiver V] [ReflQuiver W] {obj : V → W} {map} {X : V} :
    (Prefunctor.mk obj map).obj X = obj X := rfl


lemma mk_map {V W : Type*} [ReflQuiver V] [ReflQuiver W] {obj : V → W} {map} {X Y : V} {f : X ⟶ Y} :
    (Prefunctor.mk obj map).map f = map f := rfl


/-- Proving equality between reflexive prefunctors. This isn't an extensionality lemma,
  because usually you don't really want to do this. -/
theorem ext {V : Type u} [ReflQuiver.{v₁} V] {W : Type u₂} [ReflQuiver.{v₂} W]
    {F G : ReflPrefunctor V W}
    (h_obj : ∀ X, F.obj X = G.obj X)
    (h_map : ∀ (X Y : V) (f : X ⟶ Y),
      F.map f = Eq.recOn (h_obj Y).symm (Eq.recOn (h_obj X).symm (G.map f))) : F = G := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.ReflQuiver V
    W : Type u₂
    inst✝ : CategoryTheory.ReflQuiver W
    F G : CategoryTheory.ReflPrefunctor V W
    h_obj : ∀ (X : V), Eq (F.obj X) (G.obj X)
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq (F.map f) (Eq.recOn ⋯ (Eq.recOn ⋯ …
    ⊢ Eq F G
  -/
  obtain ⟨⟨F_obj⟩⟩ := F
  /-
    case mk.mk
    V : Type u
    inst✝¹ : CategoryTheory.ReflQuiver V
    W : Type u₂
    inst✝ : CategoryTheory.ReflQuiver W
    G : CategoryTheory.ReflPrefunctor V W
    F_obj : V → W
    map✝ : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (F_obj X) (F_obj Y)
    map_id✝ : ∀ (X : V), Eq ({ obj := F_obj, map := map✝ }.map (CategoryTheory.Ref …
    h_obj : ∀ (X : V), Eq ({ obj := F_obj, map := map✝, map_id := map_id✝ }.obj X) …
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq ({ obj := F_obj, map := map✝, map …
    ⊢ Eq { obj := F_obj, map := map✝, map_id := map_id✝ } G
  -/
  obtain ⟨⟨G_obj⟩⟩ := G
  /-
    case mk.mk.mk.mk
    V : Type u
    inst✝¹ : CategoryTheory.ReflQuiver V
    W : Type u₂
    inst✝ : CategoryTheory.ReflQuiver W
    F_obj : V → W
    map✝¹ : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (F_obj X) (F_obj Y)
    map_id✝¹ : ∀ (X : V), Eq ({ obj := F_obj, map := map✝¹ }.map (CategoryTheory.R …
    G_obj : V → W
    map✝ : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (G_obj X) (G_obj Y)
    map_id✝ : ∀ (X : V), Eq ({ obj := G_obj, map := map✝ }.map (CategoryTheory.Ref …
    h_obj : ∀ (X : V), Eq ({ obj := F_obj, map := map✝¹, map_id := map_id✝¹ }.obj  …
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq ({ obj := F_obj, map := map✝¹, ma …
    ⊢ Eq { obj := F_obj, map := map✝¹, map_id := map_id✝¹ } { obj := G_obj, map := …
  -/
  obtain rfl : F_obj = G_obj := (Set.eqOn_univ F_obj G_obj).mp fun _ _ ↦ h_obj _
  /-
    case mk.mk.mk.mk
    V : Type u
    inst✝¹ : CategoryTheory.ReflQuiver V
    W : Type u₂
    inst✝ : CategoryTheory.ReflQuiver W
    F_obj : V → W
    map✝¹ : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (F_obj X) (F_obj Y)
    map_id✝¹ : ∀ (X : V), Eq ({ obj := F_obj, map := map✝¹ }.map (CategoryTheory.R …
    map✝ : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (F_obj X) (F_obj Y)
    map_id✝ : ∀ (X : V), Eq ({ obj := F_obj, map := map✝ }.map (CategoryTheory.Ref …
    h_obj : ∀ (X : V), Eq ({ obj := F_obj, map := map✝¹, map_id := map_id✝¹ }.obj  …
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq ({ obj := F_obj, map := map✝¹, ma …
    ⊢ Eq { obj := F_obj, map := map✝¹, map_id := map_id✝¹ } { obj := F_obj, map := …
  -/
  congr
  /-
    case mk.mk.mk.mk.e_toPrefunctor.e_map
    V : Type u
    inst✝¹ : CategoryTheory.ReflQuiver V
    W : Type u₂
    inst✝ : CategoryTheory.ReflQuiver W
    F_obj : V → W
    map✝¹ : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (F_obj X) (F_obj Y)
    map_id✝¹ : ∀ (X : V), Eq ({ obj := F_obj, map := map✝¹ }.map (CategoryTheory.R …
    map✝ : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (F_obj X) (F_obj Y)
    map_id✝ : ∀ (X : V), Eq ({ obj := F_obj, map := map✝ }.map (CategoryTheory.Ref …
    h_obj : ∀ (X : V), Eq ({ obj := F_obj, map := map✝¹, map_id := map_id✝¹ }.obj  …
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq ({ obj := F_obj, map := map✝¹, ma …
    ⊢ Eq map✝¹ map✝
  -/
  funext X Y f
  /-
    case mk.mk.mk.mk.e_toPrefunctor.e_map.h.h.h
    V : Type u
    inst✝¹ : CategoryTheory.ReflQuiver V
    W : Type u₂
    inst✝ : CategoryTheory.ReflQuiver W
    F_obj : V → W
    map✝¹ : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (F_obj X) (F_obj Y)
    map_id✝¹ : ∀ (X : V), Eq ({ obj := F_obj, map := map✝¹ }.map (CategoryTheory.R …
    map✝ : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (F_obj X) (F_obj Y)
    map_id✝ : ∀ (X : V), Eq ({ obj := F_obj, map := map✝ }.map (CategoryTheory.Ref …
    h_obj : ∀ (X : V), Eq ({ obj := F_obj, map := map✝¹, map_id := map_id✝¹ }.obj  …
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq ({ obj := F_obj, map := map✝¹, ma …
    X Y : V
    f : Quiver.Hom X Y
    ⊢ Eq (map✝¹ f) (map✝ f)
  -/
  simpa using h_map X Y f
  /-
    🎉 no goals
  -/


/-- This may be a more useful form of `ReflPrefunctor.ext`. -/
theorem ext' {V W : Type u} [ReflQuiver.{v} V] [ReflQuiver.{v} W]
    {F G : ReflPrefunctor V W}
    (h_obj : ∀ X, F.obj X = G.obj X)
    (h_map : ∀ (X Y : V) (f : X ⟶ Y),
      F.map f = Quiver.homOfEq (G.map f) (h_obj _).symm (h_obj _).symm) : F = G := by
  /-
    V W : Type u
    inst✝¹ : CategoryTheory.ReflQuiver V
    inst✝ : CategoryTheory.ReflQuiver W
    F G : CategoryTheory.ReflPrefunctor V W
    h_obj : ∀ (X : V), Eq (F.obj X) (G.obj X)
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq (F.map f) (Quiver.homOfEq (G.map  …
    ⊢ Eq F G
  -/
  obtain ⟨Fpre, Fid⟩ := F
  /-
    case mk
    V W : Type u
    inst✝¹ : CategoryTheory.ReflQuiver V
    inst✝ : CategoryTheory.ReflQuiver W
    G : CategoryTheory.ReflPrefunctor V W
    Fpre : Prefunctor V W
    Fid : ∀ (X : V), Eq (Fpre.map (CategoryTheory.ReflQuiver.id X)) (CategoryTheor …
    h_obj : ∀ (X : V), Eq ({ toPrefunctor := Fpre, map_id := Fid }.obj X) (G.obj X)
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq ({ toPrefunctor := Fpre, map_id : …
    ⊢ Eq { toPrefunctor := Fpre, map_id := Fid } G
  -/
  obtain ⟨Gpre, Gid⟩ := G
  /-
    case mk.mk
    V W : Type u
    inst✝¹ : CategoryTheory.ReflQuiver V
    inst✝ : CategoryTheory.ReflQuiver W
    Fpre : Prefunctor V W
    Fid : ∀ (X : V), Eq (Fpre.map (CategoryTheory.ReflQuiver.id X)) (CategoryTheor …
    Gpre : Prefunctor V W
    Gid : ∀ (X : V), Eq (Gpre.map (CategoryTheory.ReflQuiver.id X)) (CategoryTheor …
    h_obj : ∀ (X : V), Eq ({ toPrefunctor := Fpre, map_id := Fid }.obj X) ({ toPre …
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq ({ toPrefunctor := Fpre, map_id : …
    ⊢ Eq { toPrefunctor := Fpre, map_id := Fid } { toPrefunctor := Gpre, map_id := …
  -/
  simp at h_obj h_map
  /-
    case mk.mk
    V W : Type u
    inst✝¹ : CategoryTheory.ReflQuiver V
    inst✝ : CategoryTheory.ReflQuiver W
    Fpre : Prefunctor V W
    Fid : ∀ (X : V), Eq (Fpre.map (CategoryTheory.ReflQuiver.id X)) (CategoryTheor …
    Gpre : Prefunctor V W
    Gid : ∀ (X : V), Eq (Gpre.map (CategoryTheory.ReflQuiver.id X)) (CategoryTheor …
    h_obj : ∀ (X : V), Eq (Fpre.obj X) (Gpre.obj X)
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq (Fpre.map f) (Quiver.homOfEq (Gpr …
    ⊢ Eq { toPrefunctor := Fpre, map_id := Fid } { toPrefunctor := Gpre, map_id := …
  -/
  obtain rfl : Fpre = Gpre := Prefunctor.ext' (V := V) (W := W) h_obj h_map
  /-
    case mk.mk
    V W : Type u
    inst✝¹ : CategoryTheory.ReflQuiver V
    inst✝ : CategoryTheory.ReflQuiver W
    Fpre : Prefunctor V W
    Fid Gid : ∀ (X : V), Eq (Fpre.map (CategoryTheory.ReflQuiver.id X)) (CategoryT …
    h_obj : ∀ (X : V), Eq (Fpre.obj X) (Fpre.obj X)
    h_map : ∀ (X Y : V) (f : Quiver.Hom X Y), Eq (Fpre.map f) (Quiver.homOfEq (Fpr …
    ⊢ Eq { toPrefunctor := Fpre, map_id := Fid } { toPrefunctor := Fpre, map_id := …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The identity morphism between reflexive quivers. -/
@[simps!]
def id (V : Type*) [ReflQuiver V] : ReflPrefunctor V V where
  __ := Prefunctor.id _
  map_id _ := rfl


instance (V : Type*) [ReflQuiver V] : Inhabited (ReflPrefunctor V V) :=
  ⟨id V⟩


/-- Composition of morphisms between reflexive quivers. -/
@[simps!]
def comp {U : Type*} [ReflQuiver U] {V : Type*} [ReflQuiver V] {W : Type*} [ReflQuiver W]
    (F : ReflPrefunctor U V) (G : ReflPrefunctor V W) : ReflPrefunctor U W where
  __ := F.toPrefunctor.comp G.toPrefunctor
                 /-
                   U : Type u_1
                   inst✝² : CategoryTheory.ReflQuiver U
                   V : Type u_2
                   inst✝¹ : CategoryTheory.ReflQuiver V
                   W : Type u_3
                   inst✝ : CategoryTheory.ReflQuiver W
                   F : CategoryTheory.ReflPrefunctor U V
                   G : CategoryTheory.ReflPrefunctor V W
                   x✝ : U
                   ⊢ Eq (__spread✝⁻⁰.map (CategoryTheory.ReflQuiver.id x✝)) (CategoryTheory.ReflQ …
                 -/
  map_id _ := by simp [F.map_id, G.map_id]
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem comp_id {U V : Type*} [ReflQuiver U] [ReflQuiver V] (F : ReflPrefunctor U V) :
    F.comp (id _) = F := rfl


@[simp]
theorem id_comp {U V : Type*} [ReflQuiver U] [ReflQuiver V] (F : ReflPrefunctor U V) :
    (id _).comp F = F := rfl


@[simp]
theorem comp_assoc {U V W Z : Type*} [ReflQuiver U] [ReflQuiver V] [ReflQuiver W] [ReflQuiver Z]
    (F : ReflPrefunctor U V) (G : ReflPrefunctor V W) (H : ReflPrefunctor W Z) :
    (F.comp G).comp H = F.comp (G.comp H) := rfl


/-- Notation for a prefunctor between reflexive quivers. -/
infixl:50 " ⥤rq " => ReflPrefunctor


/-- Notation for composition of reflexive prefunctors. -/
infixl:60 " ⋙rq " => ReflPrefunctor.comp


/-- Notation for the identity prefunctor on a reflexive quiver. -/
notation "𝟭rq" => id


theorem congr_map {U V : Type*} [Quiver U] [Quiver V] (F : U ⥤q V) {X Y : U} {f g : X ⟶ Y}
    (h : f = g) : F.map f = F.map g := congrArg F.map h


/-- A functor has an underlying refl prefunctor.-/
def Functor.toReflPrefunctor {C D} [Category C] [Category D] (F : C ⥤ D) : C ⥤rq D := { F with }


@[simp]
theorem Functor.toReflPrefunctor_toPrefunctor {C D : Cat} (F : C ⥤ D) :
    (Functor.toReflPrefunctor F).toPrefunctor = F.toPrefunctor := rfl


/-- `Vᵒᵖ` reverses the direction of all arrows of `V`. -/
instance opposite {V} [ReflQuiver V] : ReflQuiver Vᵒᵖ where
   id X := op (𝟙rq X.unop)


instance discreteReflQuiver (V : Type u) : ReflQuiver.{u+1} (Discrete V) :=
  { discreteCategory V with }



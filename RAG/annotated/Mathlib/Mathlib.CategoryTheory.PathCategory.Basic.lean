/-- A type synonym for the category of paths in a quiver.
-/
def Paths (V : Type u₁) : Type u₁ := V


instance (V : Type u₁) [Inhabited V] : Inhabited (Paths V) := ⟨(default : V)⟩


instance categoryPaths : Category.{max u₁ v₁} (Paths V) where
  Hom := fun X Y : V => Quiver.Path X Y
  id _ := Quiver.Path.nil
  comp f g := Quiver.Path.comp f g


/-- The inclusion of a quiver `V` into its path category, as a prefunctor.
-/
@[simps]
def of : V ⥤q Paths V where
  obj X := X
  map f := f.toPath


/-- To prove a property on morphisms of a path category with given source `a`, it suffices to
prove it for the identity and prove that the property is preserved under composition on the right
with length 1 paths. -/
lemma induction_fixed_source {a : Paths V} (P : ∀ {b : Paths V}, (a ⟶ b) → Prop)
    (id : P (𝟙 a))
    (comp : ∀ {u v : V} (p : a ⟶ of.obj u) (q : u ⟶ v), P p → P (p ≫ of.map q)) :
    ∀ {b : Paths V} (f : a ⟶ b), P f := by
  /-
    V : Type u₁
    inst✝ : Quiver V
    a : CategoryTheory.Paths V
    P : {b : CategoryTheory.Paths V} → Quiver.Hom a b → Prop
    id : P (CategoryTheory.CategoryStruct.id a)
    comp : ∀ {u v : V} (p : Quiver.Hom a (CategoryTheory.Paths.of.obj u)) (q : Qui …
    ⊢ ∀ {b : CategoryTheory.Paths V} (f : Quiver.Hom a b), P f
  -/
  intro _ f
  induction f with
  | nil => exact id
  | cons _ w h => exact comp _ w h


/-- To prove a property on morphisms of a path category with given target `b`, it suffices to prove
it for the identity and prove that the property is preserved under composition on the left
with length 1 paths. -/
lemma induction_fixed_target {b : Paths V} (P : ∀ {a : Paths V}, (a ⟶ b) → Prop)
    (id : P (𝟙 b))
    (comp : ∀ {u v : V} (p : of.obj v ⟶ b) (q : u ⟶ v), P p → P (of.map q ≫ p)) :
    ∀ {a : Paths V} (f : a ⟶ b), P f := by
  /-
    V : Type u₁
    inst✝ : Quiver V
    b : CategoryTheory.Paths V
    P : {a : CategoryTheory.Paths V} → Quiver.Hom a b → Prop
    id : P (CategoryTheory.CategoryStruct.id b)
    comp : ∀ {u v : V} (p : Quiver.Hom (CategoryTheory.Paths.of.obj v) b) (q : Qui …
    ⊢ ∀ {a : CategoryTheory.Paths V} (f : Quiver.Hom a b), P f
  -/
  intro a f
  /-
    V : Type u₁
    inst✝ : Quiver V
    b : CategoryTheory.Paths V
    P : {a : CategoryTheory.Paths V} → Quiver.Hom a b → Prop
    id : P (CategoryTheory.CategoryStruct.id b)
    comp : ∀ {u v : V} (p : Quiver.Hom (CategoryTheory.Paths.of.obj v) b) (q : Qui …
    a : CategoryTheory.Paths V
    f : Quiver.Hom a b
    ⊢ P f
  -/
  generalize h : f.length = k
  induction k generalizing f a with
  | zero => cases f with
    | nil => exact id
    | cons _ _ => simp at h
  | succ k h' =>
    obtain ⟨c, f, q, hq, rfl⟩ := f.eq_toPath_comp_of_length_eq_succ h
    exact comp _ _ (h' _ hq)


/-- To prove a property on morphisms of a path category, it suffices to prove it for the identity
and prove that the property is preserved under composition on the right with length 1 paths. -/
lemma induction (P : ∀ {a b : Paths V}, (a ⟶ b) → Prop)
    (id : ∀ {v : V}, P (𝟙 (of.obj v)))
    (comp : ∀ {u v w : V} (p : of.obj u ⟶ of.obj v) (q : v ⟶ w), P p → P (p ≫ of.map q)) :
    ∀ {a b : Paths V} (f : a ⟶ b), P f :=
  fun {_} ↦ induction_fixed_source _ id comp


/-- To prove a property on morphisms of a path category, it suffices to prove it for the identity
and prove that the property is preserved under composition on the left with length 1 paths. -/
lemma induction' (P : ∀ {a b : Paths V}, (a ⟶ b) → Prop)
    (id : ∀ {v : V}, P (𝟙 (of.obj v)))
    (comp : ∀ {u v w : V} (p : u ⟶ v) (q : of.obj v ⟶ of.obj w), P q → P (of.map p ≫ q)) :
    ∀ {a b : Paths V} (f : a ⟶ b), P f := by
  /-
    V : Type u₁
    inst✝ : Quiver V
    P : {a b : CategoryTheory.Paths V} → Quiver.Hom a b → Prop
    id : ∀ {v : V}, P (CategoryTheory.CategoryStruct.id (CategoryTheory.Paths.of.o …
    comp : ∀ {u v w : V} (p : Quiver.Hom u v) (q : Quiver.Hom (CategoryTheory.Path …
    ⊢ ∀ {a b : CategoryTheory.Paths V} (f : Quiver.Hom a b), P f
  -/
  intro a b
  /-
    V : Type u₁
    inst✝ : Quiver V
    P : {a b : CategoryTheory.Paths V} → Quiver.Hom a b → Prop
    id : ∀ {v : V}, P (CategoryTheory.CategoryStruct.id (CategoryTheory.Paths.of.o …
    comp : ∀ {u v w : V} (p : Quiver.Hom u v) (q : Quiver.Hom (CategoryTheory.Path …
    a b : CategoryTheory.Paths V
    ⊢ ∀ (f : Quiver.Hom a b), P f
  -/
  revert a
  /-
    V : Type u₁
    inst✝ : Quiver V
    P : {a b : CategoryTheory.Paths V} → Quiver.Hom a b → Prop
    id : ∀ {v : V}, P (CategoryTheory.CategoryStruct.id (CategoryTheory.Paths.of.o …
    comp : ∀ {u v w : V} (p : Quiver.Hom u v) (q : Quiver.Hom (CategoryTheory.Path …
    b : CategoryTheory.Paths V
    ⊢ ∀ {a : CategoryTheory.Paths V} (f : Quiver.Hom a b), P f
  -/
  exact induction_fixed_target (P := fun f ↦ P f) id (fun _ _ ↦ comp _ _)
  /-
    🎉 no goals
  -/


/-- Any prefunctor from `V` lifts to a functor from `paths V` -/
def lift {C} [Category C] (φ : V ⥤q C) : Paths V ⥤ C where
  obj := φ.obj
  map {X} {Y} f :=
    @Quiver.Path.rec V _ X (fun Y _ => φ.obj X ⟶ φ.obj Y) (𝟙 <| φ.obj X)
      (fun _ f ihp => ihp ≫ φ.map f) Y f
  map_id _ := rfl
  map_comp f g := by
    induction g with
    | nil =>
      rw [Category.comp_id]
      rfl
    | cons g' p ih =>
      have : f ≫ Quiver.Path.cons g' p = (f ≫ g').cons p := by apply Quiver.Path.comp_cons
      rw [this]
      simp only at ih ⊢
      rw [ih, Category.assoc]


@[simp]
theorem lift_nil {C} [Category C] (φ : V ⥤q C) (X : V) :
    (lift φ).map Quiver.Path.nil = 𝟙 (φ.obj X) := rfl


@[simp]
theorem lift_cons {C} [Category C] (φ : V ⥤q C) {X Y Z : V} (p : Quiver.Path X Y) (f : Y ⟶ Z) :
    (lift φ).map (p.cons f) = (lift φ).map p ≫ φ.map f := rfl


@[simp]
theorem lift_toPath {C} [Category C] (φ : V ⥤q C) {X Y : V} (f : X ⟶ Y) :
    (lift φ).map f.toPath = φ.map f := by
  /-
    V : Type u₁
    inst✝¹ : Quiver V
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    φ : Prefunctor V C
    X Y : V
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.Paths.lift φ).map f.toPath) (φ.map f)
  -/
  dsimp [Quiver.Hom.toPath, lift]
  /-
    V : Type u₁
    inst✝¹ : Quiver V
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    φ : Prefunctor V C
    X Y : V
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (φ. …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem lift_spec {C} [Category C] (φ : V ⥤q C) : of ⋙q (lift φ).toPrefunctor = φ := by
  /-
    V : Type u₁
    inst✝¹ : Quiver V
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    φ : Prefunctor V C
    ⊢ Eq (CategoryTheory.Paths.of.comp (CategoryTheory.Paths.lift φ).toPrefunctor) φ
  -/
  fapply Prefunctor.ext
    /-
      case h_obj
      V : Type u₁
      inst✝¹ : Quiver V
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      φ : Prefunctor V C
      ⊢ ∀ (X : V), Eq ((CategoryTheory.Paths.of.comp (CategoryTheory.Paths.lift φ).t …
    -/
  · rintro X
    /-
      case h_obj
      V : Type u₁
      inst✝¹ : Quiver V
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      φ : Prefunctor V C
      X : V
      ⊢ Eq ((CategoryTheory.Paths.of.comp (CategoryTheory.Paths.lift φ).toPrefunctor …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h_map
      V : Type u₁
      inst✝¹ : Quiver V
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      φ : Prefunctor V C
      ⊢ ∀ (X Y : V) (f : Quiver.Hom X Y), Eq ((CategoryTheory.Paths.of.comp (Categor …
    -/
  · rintro X Y f
    /-
      case h_map
      V : Type u₁
      inst✝¹ : Quiver V
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      φ : Prefunctor V C
      X Y : V
      f : Quiver.Hom X Y
      ⊢ Eq ((CategoryTheory.Paths.of.comp (CategoryTheory.Paths.lift φ).toPrefunctor …
    -/
    rcases φ with ⟨φo, φm⟩
    /-
      case h_map.mk
      V : Type u₁
      inst✝¹ : Quiver V
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : V
      f : Quiver.Hom X Y
      φo : V → C
      φm : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (φo X) (φo Y)
      ⊢ Eq ((CategoryTheory.Paths.of.comp (CategoryTheory.Paths.lift { obj := φo, ma …
    -/
    dsimp [lift, Quiver.Hom.toPath]
    /-
      case h_map.mk
      V : Type u₁
      inst✝¹ : Quiver V
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X Y : V
      f : Quiver.Hom X Y
      φo : V → C
      φm : {X Y : V} → Quiver.Hom X Y → Quiver.Hom (φo X) (φo Y)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (φo …
    -/
    simp only [Category.id_comp]
    /-
      🎉 no goals
    -/


theorem lift_unique {C} [Category C] (φ : V ⥤q C) (Φ : Paths V ⥤ C)
    (hΦ : of ⋙q Φ.toPrefunctor = φ) : Φ = lift φ := by
  /-
    V : Type u₁
    inst✝¹ : Quiver V
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    φ : Prefunctor V C
    Φ : CategoryTheory.Functor (CategoryTheory.Paths V) C
    hΦ : Eq (CategoryTheory.Paths.of.comp Φ.toPrefunctor) φ
    ⊢ Eq Φ (CategoryTheory.Paths.lift φ)
  -/
  subst_vars
  /-
    V : Type u₁
    inst✝¹ : Quiver V
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    Φ : CategoryTheory.Functor (CategoryTheory.Paths V) C
    ⊢ Eq Φ (CategoryTheory.Paths.lift (CategoryTheory.Paths.of.comp Φ.toPrefunctor))
  -/
  fapply Functor.ext
    /-
      case h_obj
      V : Type u₁
      inst✝¹ : Quiver V
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      Φ : CategoryTheory.Functor (CategoryTheory.Paths V) C
      ⊢ ∀ (X : CategoryTheory.Paths V), Eq (Φ.obj X) ((CategoryTheory.Paths.lift (Ca …
    -/
  · rintro X
    /-
      case h_obj
      V : Type u₁
      inst✝¹ : Quiver V
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      Φ : CategoryTheory.Functor (CategoryTheory.Paths V) C
      X : CategoryTheory.Paths V
      ⊢ Eq (Φ.obj X) ((CategoryTheory.Paths.lift (CategoryTheory.Paths.of.comp Φ.toP …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h_map
      V : Type u₁
      inst✝¹ : Quiver V
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      Φ : CategoryTheory.Functor (CategoryTheory.Paths V) C
      ⊢ autoParam (∀ (X Y : CategoryTheory.Paths V) (f : Quiver.Hom X Y), Eq (Φ.map  …
    -/
  · rintro X Y f
    /-
      case h_map
      V : Type u₁
      inst✝¹ : Quiver V
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      Φ : CategoryTheory.Functor (CategoryTheory.Paths V) C
      X Y : CategoryTheory.Paths V
      f : Quiver.Hom X Y
      ⊢ Eq (Φ.map f) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯)  …
    -/
    dsimp [lift]
    induction f with
    | nil =>
      simp only [Category.comp_id]
      apply Functor.map_id
    | cons p f' ih =>
      simp only [Category.comp_id, Category.id_comp] at ih ⊢
      -- Porting note: Had to do substitute `p.cons f'` and `f'.toPath` by their fully qualified
      -- versions in this `have` clause (elsewhere too).
      have : Φ.map (Quiver.Path.cons p f') = Φ.map p ≫ Φ.map (Quiver.Hom.toPath f') := by
        convert Functor.map_comp Φ p (Quiver.Hom.toPath f')
      rw [this, ih]


/-- Two functors out of a path category are equal when they agree on singleton paths. -/
@[ext (iff := false)]
theorem ext_functor {C} [Category C] {F G : Paths V ⥤ C} (h_obj : F.obj = G.obj)
    (h : ∀ (a b : V) (e : a ⟶ b), F.map e.toPath =
        eqToHom (congr_fun h_obj a) ≫ G.map e.toPath ≫ eqToHom (congr_fun h_obj.symm b)) :
    F = G := by
  /-
    V : Type u₁
    inst✝¹ : Quiver V
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    F G : CategoryTheory.Functor (CategoryTheory.Paths V) C
    h_obj : Eq F.obj G.obj
    h : ∀ (a b : V) (e : Quiver.Hom a b), Eq (F.map e.toPath) (CategoryTheory.Cate …
    ⊢ Eq F G
  -/
  fapply Functor.ext
    /-
      case h_obj
      V : Type u₁
      inst✝¹ : Quiver V
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      F G : CategoryTheory.Functor (CategoryTheory.Paths V) C
      h_obj : Eq F.obj G.obj
      h : ∀ (a b : V) (e : Quiver.Hom a b), Eq (F.map e.toPath) (CategoryTheory.Cate …
      ⊢ ∀ (X : CategoryTheory.Paths V), Eq (F.obj X) (G.obj X)
    -/
  · intro X
    /-
      case h_obj
      V : Type u₁
      inst✝¹ : Quiver V
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      F G : CategoryTheory.Functor (CategoryTheory.Paths V) C
      h_obj : Eq F.obj G.obj
      h : ∀ (a b : V) (e : Quiver.Hom a b), Eq (F.map e.toPath) (CategoryTheory.Cate …
      X : CategoryTheory.Paths V
      ⊢ Eq (F.obj X) (G.obj X)
    -/
    rw [h_obj]
    /-
      🎉 no goals
    -/
    /-
      case h_map
      V : Type u₁
      inst✝¹ : Quiver V
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      F G : CategoryTheory.Functor (CategoryTheory.Paths V) C
      h_obj : Eq F.obj G.obj
      h : ∀ (a b : V) (e : Quiver.Hom a b), Eq (F.map e.toPath) (CategoryTheory.Cate …
      ⊢ autoParam (∀ (X Y : CategoryTheory.Paths V) (f : Quiver.Hom X Y), Eq (F.map  …
    -/
  · intro X Y f
    /-
      case h_map
      V : Type u₁
      inst✝¹ : Quiver V
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      F G : CategoryTheory.Functor (CategoryTheory.Paths V) C
      h_obj : Eq F.obj G.obj
      h : ∀ (a b : V) (e : Quiver.Hom a b), Eq (F.map e.toPath) (CategoryTheory.Cate …
      X Y : CategoryTheory.Paths V
      f : Quiver.Hom X Y
      ⊢ Eq (F.map f) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯)  …
    -/
    induction' f with Y' Z' g e ih
      /-
        case h_map.nil
        V : Type u₁
        inst✝¹ : Quiver V
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        F G : CategoryTheory.Functor (CategoryTheory.Paths V) C
        h_obj : Eq F.obj G.obj
        h : ∀ (a b : V) (e : Quiver.Hom a b), Eq (F.map e.toPath) (CategoryTheory.Cate …
        X Y : CategoryTheory.Paths V
        ⊢ Eq (F.map Quiver.Path.nil) (CategoryTheory.CategoryStruct.comp (CategoryTheo …
      -/
    · erw [F.map_id, G.map_id, Category.id_comp, eqToHom_trans, eqToHom_refl]
      /-
        🎉 no goals
      -/
      /-
        case h_map.cons
        V : Type u₁
        inst✝¹ : Quiver V
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        F G : CategoryTheory.Functor (CategoryTheory.Paths V) C
        h_obj : Eq F.obj G.obj
        h : ∀ (a b : V) (e : Quiver.Hom a b), Eq (F.map e.toPath) (CategoryTheory.Cate …
        X Y Y' Z' : CategoryTheory.Paths V
        g : Quiver.Path X Y'
        e : Quiver.Hom Y' Z'
        ih : Eq (F.map g) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom  …
        ⊢ Eq (F.map (g.cons e)) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eq …
      -/
    · erw [F.map_comp g (Quiver.Hom.toPath e), G.map_comp g (Quiver.Hom.toPath e), ih, h]
      /-
        case h_map.cons
        V : Type u₁
        inst✝¹ : Quiver V
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        F G : CategoryTheory.Functor (CategoryTheory.Paths V) C
        h_obj : Eq F.obj G.obj
        h : ∀ (a b : V) (e : Quiver.Hom a b), Eq (F.map e.toPath) (CategoryTheory.Cate …
        X Y Y' Z' : CategoryTheory.Paths V
        g : Quiver.Path X Y'
        e : Quiver.Hom Y' Z'
        ih : Eq (F.map g) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom  …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      simp only [Category.id_comp, eqToHom_refl, eqToHom_trans_assoc, Category.assoc]
      /-
        🎉 no goals
      -/


@[simp]
theorem Prefunctor.mapPath_comp' (F : V ⥤q W) {X Y Z : Paths V} (f : X ⟶ Y) (g : Y ⟶ Z) :
    F.mapPath (f ≫ g) = (F.mapPath f).comp (F.mapPath g) :=
  Prefunctor.mapPath_comp _ _ _


/-- A path in a category can be composed to a single morphism. -/
def composePath {X : C} : ∀ {Y : C} (_ : Path X Y), X ⟶ Y
  | _, .nil => 𝟙 X
  | _, .cons p e => composePath p ≫ e


@[simp] lemma composePath_nil {X : C} : composePath (Path.nil : Path X X) = 𝟙 X := rfl


@[simp] lemma composePath_cons {X Y Z : C} (p : Path X Y) (e : Y ⟶ Z) :
  composePath (p.cons e) = composePath p ≫ e := rfl


@[simp]
theorem composePath_toPath {X Y : C} (f : X ⟶ Y) : composePath f.toPath = f := Category.id_comp _


@[simp]
theorem composePath_comp {X Y Z : C} (f : Path X Y) (g : Path Y Z) :
    composePath (f.comp g) = composePath f ≫ composePath g := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    f : Quiver.Path X Y
    g : Quiver.Path Y Z
    ⊢ Eq (CategoryTheory.composePath (f.comp g)) (CategoryTheory.CategoryStruct.co …
  -/
  induction' g with Y' Z' g e ih
    /-
      case nil
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      f : Quiver.Path X Y
      ⊢ Eq (CategoryTheory.composePath (f.comp Quiver.Path.nil)) (CategoryTheory.Cat …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      f : Quiver.Path X Y
      Y' Z' : C
      g : Quiver.Path Y Y'
      e : Quiver.Hom Y' Z'
      ih : Eq (CategoryTheory.composePath (f.comp g)) (CategoryTheory.CategoryStruct …
      ⊢ Eq (CategoryTheory.composePath (f.comp (g.cons e))) (CategoryTheory.Category …
    -/
  · simp [ih]
    /-
      🎉 no goals
    -/


@[simp]
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO get rid of `(id X : C)` somehow?
theorem composePath_id {X : Paths C} : composePath (𝟙 X) = 𝟙 (id X : C) := rfl


@[simp]
theorem composePath_comp' {X Y Z : Paths C} (f : X ⟶ Y) (g : Y ⟶ Z) :
    composePath (f ≫ g) = composePath f ≫ composePath g :=
  composePath_comp f g


/-- Composition of paths as functor from the path category of a category to the category. -/
@[simps]
def pathComposition : Paths C ⥤ C where
  obj X := X
  map f := composePath f

-- TODO: This, and what follows, should be generalized to
-- the `HomRel` for the kernel of any functor.
-- Indeed, this should be part of an equivalence between congruence relations on a category `C`
-- and full, essentially surjective functors out of `C`.

/-- The canonical relation on the path category of a category:
two paths are related if they compose to the same morphism. -/
@[simp]
def pathsHomRel : HomRel (Paths C) := fun _ _ p q =>
  (pathComposition C).map p = (pathComposition C).map q


/-- The functor from a category to the canonical quotient of its path category. -/
@[simps]
def toQuotientPaths : C ⥤ Quotient (pathsHomRel C) where
  obj X := Quotient.mk X
  map f := Quot.mk _ f.toPath
                                                            /-
                                                              C : Type u₁
                                                              inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                              X : C
                                                              ⊢ CategoryTheory.pathsHomRel C (CategoryTheory.CategoryStruct.id X).toPath (Ca …
                                                            -/
  map_id X := Quot.sound (Quotient.CompClosure.of _ _ _ (by simp))
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                                /-
                                                                  C : Type u₁
                                                                  inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                                  X✝ Y✝ Z✝ : C
                                                                  f : Quiver.Hom X✝ Y✝
                                                                  g : Quiver.Hom Y✝ Z✝
                                                                  ⊢ CategoryTheory.pathsHomRel C (CategoryTheory.CategoryStruct.comp f g).toPath …
                                                                -/
  map_comp f g := Quot.sound (Quotient.CompClosure.of _ _ _ (by simp))
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- The functor from the canonical quotient of a path category of a category
to the original category. -/
@[simps!]
def quotientPathsTo : Quotient (pathsHomRel C) ⥤ C :=
  Quotient.lift _ (pathComposition C) fun _ _ _ _ w => w


/-- The canonical quotient of the path category of a category
is equivalent to the original category. -/
def quotientPathsEquiv : Quotient (pathsHomRel C) ≌ C where
  functor := quotientPathsTo C
  inverse := toQuotientPaths C
  unitIso :=
    NatIso.ofComponents
                   /-
                     C : Type u₁
                     inst✝ : CategoryTheory.Category.{v₁, u₁} C
                     X : CategoryTheory.Quotient (CategoryTheory.pathsHomRel C)
                     ⊢ CategoryTheory.Iso ((CategoryTheory.Functor.id (CategoryTheory.Quotient (Cat …
                   -/
      (fun X => by cases X; rfl)
                            /-
                              🎉 no goals
                            -/
      (Quot.ind fun f => by
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X✝ Y✝ : CategoryTheory.Quotient (CategoryTheory.pathsHomRel C)
          f : Quiver.Hom X✝.as Y✝.as
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Category …
        -/
        apply Quot.sound
        /-
          case a
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X✝ Y✝ : CategoryTheory.Quotient (CategoryTheory.pathsHomRel C)
          f : Quiver.Hom X✝.as Y✝.as
          ⊢ CategoryTheory.Quotient.CompClosure (CategoryTheory.pathsHomRel C) (Category …
        -/
        apply Quotient.CompClosure.of
        /-
          case a.h
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          X✝ Y✝ : CategoryTheory.Quotient (CategoryTheory.pathsHomRel C)
          f : Quiver.Hom X✝.as Y✝.as
          ⊢ CategoryTheory.pathsHomRel C (CategoryTheory.CategoryStruct.comp f (Category …
        -/
        simp [Category.comp_id, Category.id_comp, pathsHomRel])
        /-
          🎉 no goals
        -/
                                                                      /-
                                                                        C : Type u₁
                                                                        inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                                        X✝ Y✝ : C
                                                                        f : Quiver.Hom X✝ Y✝
                                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.toQuotientPaths C). …
                                                                      -/
  counitIso := NatIso.ofComponents (fun _ => Iso.refl _) (fun f => by simp [Quot.liftOn_mk])
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  functor_unitIso_comp X := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : CategoryTheory.Quotient (CategoryTheory.pathsHomRel C)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.quotientPathsTo C).m …
    -/
    cases X
    simp only [pathsHomRel, pathComposition_obj, pathComposition_map, Functor.id_obj,
               quotientPathsTo_obj, Functor.comp_obj, toQuotientPaths_obj_as,
               NatIso.ofComponents_hom_app, Iso.refl_hom, quotientPathsTo_map, Category.comp_id]
    /-
      case mk
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      as✝ : CategoryTheory.Paths C
      ⊢ Eq (Quot.liftOn (CategoryTheory.CategoryStruct.id { as := as✝ }) (fun f => C …
    -/
    rfl
    /-
      🎉 no goals
    -/



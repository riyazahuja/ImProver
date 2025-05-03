/-- `pi C` gives the cartesian product of an indexed family of categories.
-/
instance pi : Category.{max w₀ v₁} (∀ i, C i) where
  Hom X Y := ∀ i, X i ⟶ Y i
  id X i := 𝟙 (X i)
  comp f g i := f i ≫ g i


/-- This provides some assistance to typeclass search in a common situation,
which otherwise fails. (Without this `CategoryTheory.Pi.has_limit_of_has_limit_comp_eval` fails.)
-/
abbrev pi' {I : Type v₁} (C : I → Type u₁) [∀ i, Category.{v₁} (C i)] : Category.{v₁} (∀ i, C i) :=
  CategoryTheory.pi C


@[simp]
theorem id_apply (X : ∀ i, C i) (i) : (𝟙 X : ∀ i, X i ⟶ X i) i = 𝟙 (X i) :=
  rfl


@[simp]
theorem comp_apply {X Y Z : ∀ i, C i} (f : X ⟶ Y) (g : Y ⟶ Z) (i) :
    (f ≫ g : ∀ i, X i ⟶ Z i) i = f i ≫ g i :=
  rfl


@[ext]
lemma ext {X Y : ∀ i, C i} {f g : X ⟶ Y} (w : ∀ i, f i = g i) : f = g :=
  funext (w ·)


/--
The evaluation functor at `i : I`, sending an `I`-indexed family of objects to the object over `i`.
-/
@[simps]
def eval (i : I) : (∀ i, C i) ⥤ C i where
  obj f := f i
  map α := α i


instance (f : J → I) : (j : J) → Category ((C ∘ f) j) := by
  /-
    I : Type w₀
    J✝ : Type w₁
    C : I → Type u₁
    inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
    J : Type w₁
    f : J → I
    ⊢ (j : J) → CategoryTheory.Category.{?u.3203, u₁} (Function.comp C f j)
  -/
  dsimp
  /-
    I : Type w₀
    J✝ : Type w₁
    C : I → Type u₁
    inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
    J : Type w₁
    f : J → I
    ⊢ (j : J) → CategoryTheory.Category.{?u.3203, u₁} (C (f j))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Pull back an `I`-indexed family of objects to a `J`-indexed family, along a function `J → I`.
-/
@[simps]
def comap (h : J → I) : (∀ i, C i) ⥤ (∀ j, C (h j)) where
  obj f i := f (h i)
  map α i := α (h i)


/-- The natural isomorphism between
pulling back a grading along the identity function,
and the identity functor. -/
@[simps]
def comapId : comap C (id : I → I) ≅ 𝟭 (∀ i, C i) where
  hom := { app := fun X => 𝟙 X }
  inv := { app := fun X => 𝟙 X }


/-- The natural isomorphism comparing between
pulling back along two successive functions, and
pulling back along their composition
-/
@[simps!]
def comapComp (f : K → J) (g : J → I) : comap C g ⋙ comap (C ∘ g) f ≅ comap C (g ∘ f) where
  hom :=
  { app := fun X b => 𝟙 (X (g (f b)))
                                   /-
                                     I : Type w₀
                                     J✝ : Type w₁
                                     C : I → Type u₁
                                     inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
                                     J : Type w₁
                                     K : Type w₂
                                     f : K → J
                                     g : J → I
                                     X Y : (i : I) → C i
                                     f' : Quiver.Hom X Y
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Pi.comap C g).comp  …
                                   -/
    naturality := fun X Y f' => by simp only [comap, Function.comp]; funext; simp }
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
  inv :=
  { app := fun X b => 𝟙 (X (g (f b)))
                                   /-
                                     I : Type w₀
                                     J✝ : Type w₁
                                     C : I → Type u₁
                                     inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
                                     J : Type w₁
                                     K : Type w₂
                                     f : K → J
                                     g : J → I
                                     X Y : (i : I) → C i
                                     f' : Quiver.Hom X Y
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pi.comap C (Function …
                                   -/
    naturality := fun X Y f' => by simp only [comap, Function.comp]; funext; simp }
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- The natural isomorphism between pulling back then evaluating, and just evaluating. -/
@[simps!]
def comapEvalIsoEval (h : J → I) (j : J) : comap C h ⋙ eval (C ∘ h) j ≅ eval C (h j) :=
                                                /-
                                                  I : Type w₀
                                                  J✝ : Type w₁
                                                  C : I → Type u₁
                                                  inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
                                                  J : Type w₁
                                                  K : Type w₂
                                                  h : J → I
                                                  j : J
                                                  ⊢ ∀ {X Y : (i : I) → C i} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStr …
                                                -/
  NatIso.ofComponents (fun _ => Iso.refl _) (by simp only [Iso.refl]; aesop_cat)
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


instance sumElimCategory : ∀ s : I ⊕ J, Category.{v₁} (Sum.elim C D s)
  | Sum.inl i => by
    /-
      I : Type w₀
      J✝ : Type w₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      J : Type w₀
      D : J → Type u₁
      inst✝ : (j : J) → CategoryTheory.Category.{v₁, u₁} (D j)
      i : I
      ⊢ CategoryTheory.Category.{v₁, u₁} (Sum.elim C D (Sum.inl i))
    -/
    dsimp
    /-
      I : Type w₀
      J✝ : Type w₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      J : Type w₀
      D : J → Type u₁
      inst✝ : (j : J) → CategoryTheory.Category.{v₁, u₁} (D j)
      i : I
      ⊢ CategoryTheory.Category.{v₁, u₁} (C i)
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  | Sum.inr j => by
    /-
      I : Type w₀
      J✝ : Type w₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      J : Type w₀
      D : J → Type u₁
      inst✝ : (j : J) → CategoryTheory.Category.{v₁, u₁} (D j)
      j : J
      ⊢ CategoryTheory.Category.{v₁, u₁} (Sum.elim C D (Sum.inr j))
    -/
    dsimp
    /-
      I : Type w₀
      J✝ : Type w₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      J : Type w₀
      D : J → Type u₁
      inst✝ : (j : J) → CategoryTheory.Category.{v₁, u₁} (D j)
      j : J
      ⊢ CategoryTheory.Category.{v₁, u₁} (D j)
    -/
    infer_instance
    /-
      🎉 no goals
    -/

/- Porting note: replaced `Sum.rec` with `match`'s per the error about
current state of code generation -/


/-- The bifunctor combining an `I`-indexed family of objects with a `J`-indexed family of objects
to obtain an `I ⊕ J`-indexed family of objects.
-/
@[simps]
def sum : (∀ i, C i) ⥤ (∀ j, D j) ⥤ ∀ s : I ⊕ J, Sum.elim C D s where
  obj X :=
    { obj := fun Y s =>
        match s with
        | .inl i => X i
        | .inr j => Y j
      map := fun {_} {_} f s =>
        match s with
        | .inl i => 𝟙 (X i)
        | .inr j => f j }
  map {X} {X'} f :=
    { app := fun Y s =>
        match s with
        | .inl i => f i
        | .inr j => 𝟙 (Y j) }


/-- An isomorphism between `I`-indexed objects gives an isomorphism between each
pair of corresponding components. -/
@[simps]
def isoApp {X Y : ∀ i, C i} (f : X ≅ Y) (i : I) : X i ≅ Y i :=
  ⟨f.hom i, f.inv i,
       /-
         I : Type w₀
         J : Type w₁
         C : I → Type u₁
         inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
         X Y : (i : I) → C i
         f : CategoryTheory.Iso X Y
         i : I
         ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.hom i) (f.inv i)) (CategoryTheory. …
       -/
       /-
         🎉 no goals
       -/
    by rw [← comp_apply, Iso.hom_inv_id, id_apply], by rw [← comp_apply, Iso.inv_hom_id, id_apply]⟩
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem isoApp_refl (X : ∀ i, C i) (i : I) : isoApp (Iso.refl X) i = Iso.refl (X i) :=
  rfl


@[simp]
theorem isoApp_symm {X Y : ∀ i, C i} (f : X ≅ Y) (i : I) : isoApp f.symm i = (isoApp f i).symm :=
  rfl


@[simp]
theorem isoApp_trans {X Y Z : ∀ i, C i} (f : X ≅ Y) (g : Y ≅ Z) (i : I) :
    isoApp (f ≪≫ g) i = isoApp f i ≪≫ isoApp g i :=
  rfl


/-- Assemble an `I`-indexed family of functors into a functor between the pi types.
-/
@[simps]
def pi (F : ∀ i, C i ⥤ D i) : (∀ i, C i) ⥤ ∀ i, D i where
  obj f i := (F i).obj (f i)
  map α i := (F i).map (α i)


/-- Similar to `pi`, but all functors come from the same category `A`
-/
@[simps]
def pi' (f : ∀ i, A ⥤ C i) : A ⥤ ∀ i, C i where
  obj a i := (f i).obj a
  map h i := (f i).map h


/-- The projections of `Functor.pi' F` are isomorphic to the functors of the family `F` -/
@[simps!]
def pi'CompEval {A : Type*} [Category A] (F : ∀ i, A ⥤ C i) (i : I) :
    pi' F ⋙ Pi.eval C i ≅ F i :=
  Iso.refl _


@[simp]
theorem eqToHom_proj {x x' : ∀ i, C i} (h : x = x') (i : I) :
    (eqToHom h : x ⟶ x') i = eqToHom (funext_iff.mp h i) := by
  /-
    I : Type w₀
    C : I → Type u₁
    inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
    x x' : (i : I) → C i
    h : Eq x x'
    i : I
    ⊢ Eq (CategoryTheory.eqToHom h i) (CategoryTheory.eqToHom ⋯)
  -/
  subst h
  /-
    I : Type w₀
    C : I → Type u₁
    inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
    x : (i : I) → C i
    i : I
    ⊢ Eq (CategoryTheory.eqToHom ⋯ i) (CategoryTheory.eqToHom ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem pi'_eval (f : ∀ i, A ⥤ C i) (i : I) : pi' f ⋙ Pi.eval C i = f i := by
  /-
    I : Type w₀
    C : I → Type u₁
    inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
    A : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} A
    f : (i : I) → CategoryTheory.Functor A (C i)
    i : I
    ⊢ Eq ((CategoryTheory.Functor.pi' f).comp (CategoryTheory.Pi.eval C i)) (f i)
  -/
  apply Functor.ext
    /-
      case h_map
      I : Type w₀
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      A : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} A
      f : (i : I) → CategoryTheory.Functor A (C i)
      i : I
      ⊢ autoParam (∀ (X Y : A) (f_1 : Quiver.Hom X Y), Eq (((CategoryTheory.Functor. …
    -/
  · intro _ _ _
    /-
      case h_map
      I : Type w₀
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      A : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} A
      f : (i : I) → CategoryTheory.Functor A (C i)
      i : I
      X✝ Y✝ : A
      f✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq (((CategoryTheory.Functor.pi' f).comp (CategoryTheory.Pi.eval C i)).map f …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case h_obj
      I : Type w₀
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      A : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} A
      f : (i : I) → CategoryTheory.Functor A (C i)
      i : I
      ⊢ ∀ (X : A), Eq (((CategoryTheory.Functor.pi' f).comp (CategoryTheory.Pi.eval  …
    -/
  · intro _
    /-
      case h_obj
      I : Type w₀
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      A : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} A
      f : (i : I) → CategoryTheory.Functor A (C i)
      i : I
      X✝ : A
      ⊢ Eq (((CategoryTheory.Functor.pi' f).comp (CategoryTheory.Pi.eval C i)).obj X …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Two functors to a product category are equal iff they agree on every coordinate. -/
theorem pi_ext (f f' : A ⥤ ∀ i, C i) (h : ∀ i, f ⋙ (Pi.eval C i) = f' ⋙ (Pi.eval C i)) :
    f = f' := by
  /-
    I : Type w₀
    C : I → Type u₁
    inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
    A : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} A
    f f' : CategoryTheory.Functor A ((i : I) → C i)
    h : ∀ (i : I), Eq (f.comp (CategoryTheory.Pi.eval C i)) (f'.comp (CategoryTheo …
    ⊢ Eq f f'
  -/
  apply Functor.ext; rotate_left
    /-
      case h_obj
      I : Type w₀
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      A : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} A
      f f' : CategoryTheory.Functor A ((i : I) → C i)
      h : ∀ (i : I), Eq (f.comp (CategoryTheory.Pi.eval C i)) (f'.comp (CategoryTheo …
      ⊢ ∀ (X : A), Eq (f.obj X) (f'.obj X)
    -/
  · intro X
    /-
      case h_obj
      I : Type w₀
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      A : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} A
      f f' : CategoryTheory.Functor A ((i : I) → C i)
      h : ∀ (i : I), Eq (f.comp (CategoryTheory.Pi.eval C i)) (f'.comp (CategoryTheo …
      X : A
      ⊢ Eq (f.obj X) (f'.obj X)
    -/
    ext i
    /-
      case h_obj.h
      I : Type w₀
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      A : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} A
      f f' : CategoryTheory.Functor A ((i : I) → C i)
      h : ∀ (i : I), Eq (f.comp (CategoryTheory.Pi.eval C i)) (f'.comp (CategoryTheo …
      X : A
      i : I
      ⊢ Eq (f.obj X i) (f'.obj X i)
    -/
    specialize h i
    /-
      case h_obj.h
      I : Type w₀
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      A : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} A
      f f' : CategoryTheory.Functor A ((i : I) → C i)
      X : A
      i : I
      h : Eq (f.comp (CategoryTheory.Pi.eval C i)) (f'.comp (CategoryTheory.Pi.eval  …
      ⊢ Eq (f.obj X i) (f'.obj X i)
    -/
    have := congr_obj h X
    /-
      case h_obj.h
      I : Type w₀
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      A : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} A
      f f' : CategoryTheory.Functor A ((i : I) → C i)
      X : A
      i : I
      h : Eq (f.comp (CategoryTheory.Pi.eval C i)) (f'.comp (CategoryTheory.Pi.eval  …
      this : Eq ((f.comp (CategoryTheory.Pi.eval C i)).obj X) ((f'.comp (CategoryThe …
      ⊢ Eq (f.obj X i) (f'.obj X i)
    -/
    simpa
    /-
      🎉 no goals
    -/
    /-
      case h_map
      I : Type w₀
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      A : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} A
      f f' : CategoryTheory.Functor A ((i : I) → C i)
      h : ∀ (i : I), Eq (f.comp (CategoryTheory.Pi.eval C i)) (f'.comp (CategoryTheo …
      ⊢ autoParam (∀ (X Y : A) (f_1 : Quiver.Hom X Y), Eq (f.map f_1) (CategoryTheor …
    -/
  · intro X Y g
    /-
      case h_map
      I : Type w₀
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      A : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} A
      f f' : CategoryTheory.Functor A ((i : I) → C i)
      h : ∀ (i : I), Eq (f.comp (CategoryTheory.Pi.eval C i)) (f'.comp (CategoryTheo …
      X Y : A
      g : Quiver.Hom X Y
      ⊢ Eq (f.map g) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯)  …
    -/
    dsimp
    /-
      case h_map
      I : Type w₀
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      A : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} A
      f f' : CategoryTheory.Functor A ((i : I) → C i)
      h : ∀ (i : I), Eq (f.comp (CategoryTheory.Pi.eval C i)) (f'.comp (CategoryTheo …
      X Y : A
      g : Quiver.Hom X Y
      ⊢ Eq (f.map g) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯)  …
    -/
    funext i
    /-
      case h_map.h
      I : Type w₀
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      A : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} A
      f f' : CategoryTheory.Functor A ((i : I) → C i)
      h : ∀ (i : I), Eq (f.comp (CategoryTheory.Pi.eval C i)) (f'.comp (CategoryTheo …
      X Y : A
      g : Quiver.Hom X Y
      i : I
      ⊢ Eq (f.map g i) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯ …
    -/
    specialize h i
    /-
      case h_map.h
      I : Type w₀
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      A : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} A
      f f' : CategoryTheory.Functor A ((i : I) → C i)
      h✝ : ∀ (i : I), Eq (f.comp (CategoryTheory.Pi.eval C i)) (f'.comp (CategoryThe …
      X Y : A
      g : Quiver.Hom X Y
      i : I
      h : Eq (f.comp (CategoryTheory.Pi.eval C i)) (f'.comp (CategoryTheory.Pi.eval  …
      ⊢ Eq (f.map g i) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯ …
    -/
    have := congr_hom h g
    /-
      case h_map.h
      I : Type w₀
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      A : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} A
      f f' : CategoryTheory.Functor A ((i : I) → C i)
      h✝ : ∀ (i : I), Eq (f.comp (CategoryTheory.Pi.eval C i)) (f'.comp (CategoryThe …
      X Y : A
      g : Quiver.Hom X Y
      i : I
      h : Eq (f.comp (CategoryTheory.Pi.eval C i)) (f'.comp (CategoryTheory.Pi.eval  …
      this : Eq ((f.comp (CategoryTheory.Pi.eval C i)).map g) (CategoryTheory.Catego …
      ⊢ Eq (f.map g i) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯ …
    -/
    simpa
    /-
      🎉 no goals
    -/


/-- Assemble an `I`-indexed family of natural transformations into a single natural transformation.
-/
@[simps!]
def pi (α : ∀ i, F i ⟶ G i) : Functor.pi F ⟶ Functor.pi G where
  app f i := (α i).app (f i)


/-- Assemble an `I`-indexed family of natural transformations into a single natural transformation.
-/
@[simps]
def pi' {E : Type*} [Category E] {F G : E ⥤ ∀ i, C i}
    (τ : ∀ i, F ⋙ Pi.eval C i ⟶ G ⋙ Pi.eval C i) : F ⟶ G where
  app := fun X i => (τ i).app X
  naturality _ _ f := by
    /-
      I : Type w₀
      J : Type w₁
      C : I → Type u₁
      inst✝² : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      D : I → Type u₂
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₂, u₂} (D i)
      F✝ G✝ : (i : I) → CategoryTheory.Functor (C i) (D i)
      E : Type u_1
      inst✝ : CategoryTheory.Category.{?u.282187, u_1} E
      F G : CategoryTheory.Functor E ((i : I) → C i)
      τ : (i : I) → Quiver.Hom (F.comp (CategoryTheory.Pi.eval C i)) (G.comp (Catego …
      x✝¹ x✝ : E
      f : Quiver.Hom x✝¹ x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun X i => (τ i).app X) x …
    -/
    ext i
    /-
      case w
      I : Type w₀
      J : Type w₁
      C : I → Type u₁
      inst✝² : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      D : I → Type u₂
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₂, u₂} (D i)
      F✝ G✝ : (i : I) → CategoryTheory.Functor (C i) (D i)
      E : Type u_1
      inst✝ : CategoryTheory.Category.{?u.282187, u_1} E
      F G : CategoryTheory.Functor E ((i : I) → C i)
      τ : (i : I) → Quiver.Hom (F.comp (CategoryTheory.Pi.eval C i)) (G.comp (Catego …
      x✝¹ x✝ : E
      f : Quiver.Hom x✝¹ x✝
      i : I
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun X i => (τ i).app X) x …
    -/
    exact (τ i).naturality f
    /-
      🎉 no goals
    -/


/-- Assemble an `I`-indexed family of natural isomorphisms into a single natural isomorphism.
-/
@[simps]
def pi (e : ∀ i, F i ≅ G i) : Functor.pi F ≅ Functor.pi G where
  hom := NatTrans.pi (fun i => (e i).hom)
  inv := NatTrans.pi (fun i => (e i).inv)


/-- Assemble an `I`-indexed family of natural isomorphisms into a single natural isomorphism.
-/
@[simps]
def pi' {E : Type*} [Category E] {F G : E ⥤ ∀ i, C i}
    (e : ∀ i, F ⋙ Pi.eval C i ≅ G ⋙ Pi.eval C i) : F ≅ G where
  hom := NatTrans.pi' (fun i => (e i).hom)
  inv := NatTrans.pi' (fun i => (e i).inv)


lemma isIso_pi_iff {X Y : ∀ i, C i} (f : X ⟶ Y) :
    IsIso f ↔ ∀ i, IsIso (f i) := by
  /-
    I : Type w₀
    C : I → Type u₁
    inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
    X Y : (i : I) → C i
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.IsIso f) (∀ (i : I), CategoryTheory.IsIso (f i))
  -/
  constructor
    /-
      case mp
      I : Type w₀
      C : I → Type u₁
      inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      X Y : (i : I) → C i
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.IsIso f → ∀ (i : I), CategoryTheory.IsIso (f i)
    -/
  · intro _ i
    /-
      case mp
      I : Type w₀
      C : I → Type u₁
      inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      X Y : (i : I) → C i
      f : Quiver.Hom X Y
      a✝ : CategoryTheory.IsIso f
      i : I
      ⊢ CategoryTheory.IsIso (f i)
    -/
    exact (Pi.isoApp (asIso f) i).isIso_hom
    /-
      🎉 no goals
    -/
    /-
      case mpr
      I : Type w₀
      C : I → Type u₁
      inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      X Y : (i : I) → C i
      f : Quiver.Hom X Y
      ⊢ (∀ (i : I), CategoryTheory.IsIso (f i)) → CategoryTheory.IsIso f
    -/
  · intro
    /-
      case mpr
      I : Type w₀
      C : I → Type u₁
      inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      X Y : (i : I) → C i
      f : Quiver.Hom X Y
      a✝ : ∀ (i : I), CategoryTheory.IsIso (f i)
      ⊢ CategoryTheory.IsIso f
    -/
    exact ⟨fun i => inv (f i), by aesop_cat, by aesop_cat⟩
    /-
      🎉 no goals
    -/


/-- For a family of categories `C i` indexed by `I`, an equality `i = j` in `I` induces
an equivalence `C i ≌ C j`. -/
                                                               /-
                                                                 I : Type w₀
                                                                 J : Type w₁
                                                                 C : I → Type u₁
                                                                 inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
                                                                 i j : I
                                                                 h : Eq i j
                                                                 ⊢ CategoryTheory.Equivalence (C i) (C j)
                                                               -/
def Pi.eqToEquivalence {i j : I} (h : i = j) : C i ≌ C j := by subst h; rfl
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- When `i = j`, projections `Pi.eval C i` and `Pi.eval C j` are related by the equivalence
`Pi.eqToEquivalence C h : C i ≌ C j`. -/
@[simps!]
def Pi.evalCompEqToEquivalenceFunctor {i j : I} (h : i = j) :
    Pi.eval C i ⋙ (Pi.eqToEquivalence C h).functor ≅
      Pi.eval C j :=
              /-
                I : Type w₀
                J : Type w₁
                C : I → Type u₁
                inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
                i j : I
                h : Eq i j
                ⊢ Eq ((CategoryTheory.Pi.eval C i).comp (CategoryTheory.Pi.eqToEquivalence C h …
              -/
  eqToIso (by subst h; rfl)
                       /-
                         🎉 no goals
                       -/


/-- The equivalences given by `Pi.eqToEquivalence` are compatible with reindexing. -/
@[simps!]
def Pi.eqToEquivalenceFunctorIso (f : J → I) {i' j' : J} (h : i' = j') :
    (Pi.eqToEquivalence C (congr_arg f h)).functor ≅
      (Pi.eqToEquivalence (fun i' => C (f i')) h).functor :=
              /-
                I : Type w₀
                J : Type w₁
                C : I → Type u₁
                inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
                f : J → I
                i' j' : J
                h : Eq i' j'
                ⊢ Eq (CategoryTheory.Pi.eqToEquivalence C ⋯).functor (CategoryTheory.Pi.eqToEq …
              -/
  eqToIso (by subst h; rfl)
                       /-
                         🎉 no goals
                       -/


/-- Reindexing a family of categories gives equivalent `Pi` categories. -/
@[simps]
noncomputable def Pi.equivalenceOfEquiv (e : J ≃ I) :
    (∀ j, C (e j)) ≌ (∀ i, C i) where
  functor := Functor.pi' (fun i => Pi.eval _ (e.symm i) ⋙
                              /-
                                I : Type w₀
                                J : Type w₁
                                C : I → Type u₁
                                inst✝ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
                                e : Equiv J I
                                i : I
                                ⊢ Eq (e (e.symm i)) i
                              -/
    (Pi.eqToEquivalence C (by simp)).functor)
                              /-
                                🎉 no goals
                              -/
  inverse := Functor.pi' (fun i' => Pi.eval _ (e i'))
  unitIso := NatIso.pi' (fun i' => Functor.leftUnitor _ ≪≫
    (Pi.evalCompEqToEquivalenceFunctor (fun j => C (e j)) (e.symm_apply_apply i')).symm ≪≫
    isoWhiskerLeft _ ((Pi.eqToEquivalenceFunctorIso C e (e.symm_apply_apply i')).symm) ≪≫
    (Functor.pi'CompEval _ _).symm ≪≫ isoWhiskerLeft _ (Functor.pi'CompEval _ _).symm ≪≫
    (Functor.associator _ _ _).symm)
  counitIso := NatIso.pi' (fun i => (Functor.associator _ _ _).symm ≪≫
    isoWhiskerRight (Functor.pi'CompEval _ _) _ ≪≫
    Pi.evalCompEqToEquivalenceFunctor C (e.apply_symm_apply i) ≪≫
    (Functor.leftUnitor _).symm)


/-- A product of categories indexed by `Option J` identifies to a binary product. -/
@[simps]
def Pi.optionEquivalence (C' : Option J → Type u₁) [∀ i, Category.{v₁} (C' i)] :
    (∀ i, C' i) ≌ C' none × (∀ (j : J), C' (some j)) where
  functor := Functor.prod' (Pi.eval C' none)
    (Functor.pi' (fun i => (Pi.eval _ (some i))))
  inverse := Functor.pi' (fun i => match i with
    | none => Prod.fst _ _
    | some i => Prod.snd _ _ ⋙ (Pi.eval _ i))
  unitIso := NatIso.pi' (fun i => match i with
    | none => Iso.refl _
    | some _ => Iso.refl _)
                  /-
                    I : Type w₀
                    J : Type w₁
                    C : I → Type u₁
                    inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
                    C' : Option J → Type u₁
                    inst✝ : (i : Option J) → CategoryTheory.Category.{v₁, u₁} (C' i)
                    ⊢ CategoryTheory.Iso ((CategoryTheory.Functor.pi' fun i => CategoryTheory.Pi.o …
                  -/
  counitIso := by exact Iso.refl _
                  /-
                    🎉 no goals
                  -/


/-- Assemble an `I`-indexed family of equivalences of categories
into a single equivalence. -/
@[simps]
def pi (E : ∀ i, C i ≌ D i) : (∀ i, C i) ≌ (∀ i, D i) where
  functor := Functor.pi (fun i => (E i).functor)
  inverse := Functor.pi (fun i => (E i).inverse)
  unitIso := NatIso.pi (fun i => (E i).unitIso)
  counitIso := NatIso.pi (fun i => (E i).counitIso)


instance (F : ∀ i, C i ⥤ D i) [∀ i, (F i).IsEquivalence] :
    (Functor.pi F).IsEquivalence :=
  (pi (fun i => (F i).asEquivalence)).isEquivalence_functor



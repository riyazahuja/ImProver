/-- The category of `T`-structured arrows with domain `S : D` (here `T : C ⥤ D`),
has as its objects `D`-morphisms of the form `S ⟶ T Y`, for some `Y : C`,
and morphisms `C`-morphisms `Y ⟶ Y'` making the obvious triangle commute.
-/
-- We explicitly come from `PUnit.{1}` here to obtain the correct universe for morphisms of
-- structured arrows.
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): linter not ported yet
-- @[nolint has_nonempty_instance]
def StructuredArrow (S : D) (T : C ⥤ D) :=
  Comma (Functor.fromPUnit.{0} S) T

-- Porting note: not found by inferInstance

instance (S : D) (T : C ⥤ D) : Category (StructuredArrow S T) := commaCategory


/-- The obvious projection functor from structured arrows. -/
@[simps!]
def proj (S : D) (T : C ⥤ D) : StructuredArrow S T ⥤ C :=
  Comma.snd _ _


@[ext]
lemma hom_ext {X Y : StructuredArrow S T} (f g : X ⟶ Y) (h : f.right = g.right) : f = g :=
  CommaMorphism.ext (Subsingleton.elim _ _) h


@[simp]
theorem hom_eq_iff {X Y : StructuredArrow S T} (f g : X ⟶ Y) : f = g ↔ f.right = g.right :=
              /-
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                D : Type u₂
                inst✝ : CategoryTheory.Category.{v₂, u₂} D
                S : D
                T : CategoryTheory.Functor C D
                X Y : CategoryTheory.StructuredArrow S T
                f g : Quiver.Hom X Y
                h : Eq f g
                ⊢ Eq f.right g.right
              -/
  ⟨fun h ↦ by rw [h], hom_ext _ _⟩
              /-
                🎉 no goals
              -/


/-- Construct a structured arrow from a morphism. -/
def mk (f : S ⟶ T.obj Y) : StructuredArrow S T :=
  ⟨⟨⟨⟩⟩, Y, f⟩


@[simp]
theorem mk_left (f : S ⟶ T.obj Y) : (mk f).left = ⟨⟨⟩⟩ :=
  rfl


@[simp]
theorem mk_right (f : S ⟶ T.obj Y) : (mk f).right = Y :=
  rfl


@[simp]
theorem mk_hom_eq_self (f : S ⟶ T.obj Y) : (mk f).hom = f :=
  rfl


@[reassoc (attr := simp)]
theorem w {A B : StructuredArrow S T} (f : A ⟶ B) : A.hom ≫ T.map f.right = B.hom := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S : D
    T : CategoryTheory.Functor C D
    A B : CategoryTheory.StructuredArrow S T
    f : Quiver.Hom A B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp A.hom (T.map f.right)) B.hom
  -/
  have := f.w; aesop_cat
               /-
                 🎉 no goals
               -/


@[simp]
theorem comp_right {X Y Z : StructuredArrow S T} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).right = f.right ≫ g.right := rfl


@[simp]
theorem id_right (X : StructuredArrow S T) : (𝟙 X : X ⟶ X).right = 𝟙 X.right := rfl


@[simp]
theorem eqToHom_right {X Y : StructuredArrow S T} (h : X = Y) :
                                    /-
                                      C : Type u₁
                                      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                      D : Type u₂
                                      inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                      S S' S'' : D
                                      Y✝ Y' Y'' : C
                                      T T' : CategoryTheory.Functor C D
                                      X Y : CategoryTheory.StructuredArrow S T
                                      h : Eq X Y
                                      ⊢ Eq X.right Y.right
                                    -/
    (eqToHom h).right = eqToHom (by rw [h]) := by
                                    /-
                                      🎉 no goals
                                    -/
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S : D
    T : CategoryTheory.Functor C D
    X Y : CategoryTheory.StructuredArrow S T
    h : Eq X Y
    ⊢ Eq (CategoryTheory.eqToHom h).right (CategoryTheory.eqToHom ⋯)
  -/
  subst h
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S : D
    T : CategoryTheory.Functor C D
    X : CategoryTheory.StructuredArrow S T
    ⊢ Eq (CategoryTheory.eqToHom ⋯).right (CategoryTheory.eqToHom ⋯)
  -/
  simp only [eqToHom_refl, id_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem left_eq_id {X Y : StructuredArrow S T} (f : X ⟶ Y) : f.left = 𝟙 X.left := rfl


/-- To construct a morphism of structured arrows,
we need a morphism of the objects underlying the target,
and to check that the triangle commutes.
-/
@[simps]
def homMk {f f' : StructuredArrow S T} (g : f.right ⟶ f'.right)
    (w : f.hom ≫ T.map g = f'.hom := by aesop_cat) : f ⟶ f' where
  left := 𝟙 f.left
  right := g
  w := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      S S' S'' : D
      Y Y' Y'' : C
      T T' : CategoryTheory.Functor C D
      f f' : CategoryTheory.StructuredArrow S T
      g : Quiver.Hom f.right f'.right
      w : autoParam (Eq (CategoryTheory.CategoryStruct.comp f.hom (T.map g)) f'.hom) …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.fromPUnit S) …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      S S' S'' : D
      Y Y' Y'' : C
      T T' : CategoryTheory.Functor C D
      f f' : CategoryTheory.StructuredArrow S T
      g : Quiver.Hom f.right f'.right
      w : autoParam (Eq (CategoryTheory.CategoryStruct.comp f.hom (T.map g)) f'.hom) …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id S)  …
    -/
    simpa using w.symm
    /-
      🎉 no goals
    -/

/- Porting note: it appears the simp lemma is not getting generated but the linter
picks up on it (seems like a bug). Either way simp solves it. -/

theorem homMk_surjective {f f' : StructuredArrow S T} (φ : f ⟶ f') :
    ∃ (ψ : f.right ⟶ f'.right) (hψ : f.hom ≫ T.map ψ = f'.hom),
      φ = StructuredArrow.homMk ψ hψ :=
  ⟨φ.right, StructuredArrow.w φ, rfl⟩


/-- Given a structured arrow `X ⟶ T(Y)`, and an arrow `Y ⟶ Y'`, we can construct a morphism of
    structured arrows given by `(X ⟶ T(Y)) ⟶ (X ⟶ T(Y) ⟶ T(Y'))`. -/
@[simps]
def homMk' (f : StructuredArrow S T) (g : f.right ⟶ Y') : f ⟶ mk (f.hom ≫ T.map g) where
  left := 𝟙 _
  right := g


                                                                               /-
                                                                                 C : Type u₁
                                                                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                 D : Type u₂
                                                                                 inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                                 S S' S'' : D
                                                                                 Y Y' Y'' : C
                                                                                 T T' : CategoryTheory.Functor C D
                                                                                 f : CategoryTheory.StructuredArrow S T
                                                                                 ⊢ Eq f (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.comp  …
                                                                               -/
lemma homMk'_id (f : StructuredArrow S T) : homMk' f (𝟙 f.right) = eqToHom (by aesop_cat) := by
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S : D
    T : CategoryTheory.Functor C D
    f : CategoryTheory.StructuredArrow S T
    ⊢ Eq (f.homMk' (CategoryTheory.CategoryStruct.id f.right)) (CategoryTheory.eqT …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S : D
    T : CategoryTheory.Functor C D
    f : CategoryTheory.StructuredArrow S T
    ⊢ Eq (f.homMk' (CategoryTheory.CategoryStruct.id f.right)).right (CategoryTheo …
  -/
  simp [eqToHom_right]
  /-
    🎉 no goals
  -/


                                                                         /-
                                                                           C : Type u₁
                                                                           inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                           D : Type u₂
                                                                           inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                           S S' S'' : D
                                                                           Y Y' Y'' : C
                                                                           T T' : CategoryTheory.Functor C D
                                                                           f : Quiver.Hom S (T.obj Y)
                                                                           ⊢ Eq (CategoryTheory.StructuredArrow.mk f) (CategoryTheory.StructuredArrow.mk  …
                                                                         -/
lemma homMk'_mk_id (f : S ⟶ T.obj Y) : homMk' (mk f) (𝟙 Y) = eqToHom (by aesop_cat) :=
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
  homMk'_id _


lemma homMk'_comp (f : StructuredArrow S T) (g : f.right ⟶ Y') (g' : Y' ⟶ Y'') :
                                                                                    /-
                                                                                      C : Type u₁
                                                                                      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                      D : Type u₂
                                                                                      inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                                      S S' S'' : D
                                                                                      Y Y' Y'' : C
                                                                                      T T' : CategoryTheory.Functor C D
                                                                                      f : CategoryTheory.StructuredArrow S T
                                                                                      g : Quiver.Hom f.right Y'
                                                                                      g' : Quiver.Hom Y' Y''
                                                                                      ⊢ Eq (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.comp (C …
                                                                                    -/
    homMk' f (g ≫ g') = homMk' f g ≫ homMk' (mk (f.hom ≫ T.map g)) g' ≫ eqToHom (by simp) := by
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S : D
    Y' Y'' : C
    T : CategoryTheory.Functor C D
    f : CategoryTheory.StructuredArrow S T
    g : Quiver.Hom f.right Y'
    g' : Quiver.Hom Y' Y''
    ⊢ Eq (f.homMk' (CategoryTheory.CategoryStruct.comp g g')) (CategoryTheory.Cate …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S : D
    Y' Y'' : C
    T : CategoryTheory.Functor C D
    f : CategoryTheory.StructuredArrow S T
    g : Quiver.Hom f.right Y'
    g' : Quiver.Hom Y' Y''
    ⊢ Eq (f.homMk' (CategoryTheory.CategoryStruct.comp g g')).right (CategoryTheor …
  -/
  simp [eqToHom_right]
  /-
    🎉 no goals
  -/


lemma homMk'_mk_comp (f : S ⟶ T.obj Y) (g : Y ⟶ Y') (g' : Y' ⟶ Y'') :
                                                                                          /-
                                                                                            C : Type u₁
                                                                                            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                            D : Type u₂
                                                                                            inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                                            S S' S'' : D
                                                                                            Y Y' Y'' : C
                                                                                            T T' : CategoryTheory.Functor C D
                                                                                            f : Quiver.Hom S (T.obj Y)
                                                                                            g : Quiver.Hom Y Y'
                                                                                            g' : Quiver.Hom Y' Y''
                                                                                            ⊢ Eq (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.comp (C …
                                                                                          -/
    homMk' (mk f) (g ≫ g') = homMk' (mk f) g ≫ homMk' (mk (f ≫ T.map g)) g' ≫ eqToHom (by simp) :=
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
  homMk'_comp _ _ _


/-- Variant of `homMk'` where both objects are applications of `mk`. -/
@[simps]
def mkPostcomp (f : S ⟶ T.obj Y) (g : Y ⟶ Y') : mk f ⟶ mk (f ≫ T.map g) where
  left := 𝟙 _
  right := g


                                                                         /-
                                                                           C : Type u₁
                                                                           inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                           D : Type u₂
                                                                           inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                           S S' S'' : D
                                                                           Y Y' Y'' : C
                                                                           T T' : CategoryTheory.Functor C D
                                                                           f : Quiver.Hom S (T.obj Y)
                                                                           ⊢ Eq (CategoryTheory.StructuredArrow.mk f) (CategoryTheory.StructuredArrow.mk  …
                                                                         -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
lemma mkPostcomp_id (f : S ⟶ T.obj Y) : mkPostcomp f (𝟙 Y) = eqToHom (by aesop_cat) := by aesop_cat
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/

lemma mkPostcomp_comp (f : S ⟶ T.obj Y) (g : Y ⟶ Y') (g' : Y' ⟶ Y'') :
                                                                                       /-
                                                                                         C : Type u₁
                                                                                         inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                         D : Type u₂
                                                                                         inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                                         S S' S'' : D
                                                                                         Y Y' Y'' : C
                                                                                         T T' : CategoryTheory.Functor C D
                                                                                         f : Quiver.Hom S (T.obj Y)
                                                                                         g : Quiver.Hom Y Y'
                                                                                         g' : Quiver.Hom Y' Y''
                                                                                         ⊢ Eq (CategoryTheory.StructuredArrow.mk (CategoryTheory.CategoryStruct.comp (C …
                                                                                       -/
    mkPostcomp f (g ≫ g') = mkPostcomp f g ≫ mkPostcomp (f ≫ T.map g) g' ≫ eqToHom (by simp) := by
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S : D
    Y Y' Y'' : C
    T : CategoryTheory.Functor C D
    f : Quiver.Hom S (T.obj Y)
    g : Quiver.Hom Y Y'
    g' : Quiver.Hom Y' Y''
    ⊢ Eq (CategoryTheory.StructuredArrow.mkPostcomp f (CategoryTheory.CategoryStru …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


/-- To construct an isomorphism of structured arrows,
we need an isomorphism of the objects underlying the target,
and to check that the triangle commutes.
-/
@[simps!]
def isoMk {f f' : StructuredArrow S T} (g : f.right ≅ f'.right)
    (w : f.hom ≫ T.map g.hom = f'.hom := by aesop_cat) :
    f ≅ f' :=
                           /-
                             C : Type u₁
                             inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                             D : Type u₂
                             inst✝ : CategoryTheory.Category.{v₂, u₂} D
                             S S' S'' : D
                             Y Y' Y'' : C
                             T T' : CategoryTheory.Functor C D
                             f f' : CategoryTheory.StructuredArrow S T
                             g : CategoryTheory.Iso f.right f'.right
                             w : autoParam (Eq (CategoryTheory.CategoryStruct.comp f.hom (T.map g.hom)) f'. …
                             ⊢ Eq f.left f'.left
                           -/
                           /-
                             🎉 no goals
                           -/
  Comma.isoMk (eqToIso (by ext)) g (by simpa using w.symm)
                                       /-
                                         🎉 no goals
                                       -/

/- Porting note: it appears the simp lemma is not getting generated but the linter
picks up on it. Either way simp solves these. -/

theorem ext {A B : StructuredArrow S T} (f g : A ⟶ B) : f.right = g.right → f = g :=
  CommaMorphism.ext (Subsingleton.elim _ _)


theorem ext_iff {A B : StructuredArrow S T} (f g : A ⟶ B) : f = g ↔ f.right = g.right :=
  ⟨fun h => h ▸ rfl, ext f g⟩


instance proj_faithful : (proj S T).Faithful where
  map_injective {_ _} := ext


/-- The converse of this is true with additional assumptions, see `mono_iff_mono_right`. -/
theorem mono_of_mono_right {A B : StructuredArrow S T} (f : A ⟶ B) [h : Mono f.right] : Mono f :=
  (proj S T).mono_of_mono_map h


theorem epi_of_epi_right {A B : StructuredArrow S T} (f : A ⟶ B) [h : Epi f.right] : Epi f :=
  (proj S T).epi_of_epi_map h


instance mono_homMk {A B : StructuredArrow S T} (f : A.right ⟶ B.right) (w) [h : Mono f] :
    Mono (homMk f w) :=
  (proj S T).mono_of_mono_map h


instance epi_homMk {A B : StructuredArrow S T} (f : A.right ⟶ B.right) (w) [h : Epi f] :
    Epi (homMk f w) :=
  (proj S T).epi_of_epi_map h


/-- Eta rule for structured arrows. Prefer `StructuredArrow.eta` for rewriting, since equality of
    objects tends to cause problems. -/
theorem eq_mk (f : StructuredArrow S T) : f = mk f.hom :=
  rfl


/-- Eta rule for structured arrows. -/
@[simps!]
def eta (f : StructuredArrow S T) : f ≅ mk f.hom :=
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S S' S'' : D
    Y Y' Y'' : C
    T T' : CategoryTheory.Functor C D
    f : CategoryTheory.StructuredArrow S T
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom (T.map (CategoryTheory.Iso.refl …
  -/
  isoMk (Iso.refl _)
  /-
    🎉 no goals
  -/

/- Porting note: it appears the simp lemma is not getting generated but the linter
picks up on it. Either way simp solves these. -/

lemma mk_surjective (f : StructuredArrow S T) :
    ∃ (Y : C) (g : S ⟶ T.obj Y), f = mk g :=
  ⟨_, _, eq_mk f⟩


/-- A morphism between source objects `S ⟶ S'`
contravariantly induces a functor between structured arrows,
`StructuredArrow S' T ⥤ StructuredArrow S T`.

Ideally this would be described as a 2-functor from `D`
(promoted to a 2-category with equations as 2-morphisms)
to `Cat`.
-/
@[simps!]
def map (f : S ⟶ S') : StructuredArrow S' T ⥤ StructuredArrow S T :=
  Comma.mapLeft _ ((Functor.const _).map f)


@[simp]
theorem map_mk {f : S' ⟶ T.obj Y} (g : S ⟶ S') : (map g).obj (mk f) = mk (g ≫ f) :=
  rfl


@[simp]
theorem map_id {f : StructuredArrow S T} : (map (𝟙 S)).obj f = f := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S : D
    T : CategoryTheory.Functor C D
    f : CategoryTheory.StructuredArrow S T
    ⊢ Eq ((CategoryTheory.StructuredArrow.map (CategoryTheory.CategoryStruct.id S) …
  -/
  rw [eq_mk f]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S : D
    T : CategoryTheory.Functor C D
    f : CategoryTheory.StructuredArrow S T
    ⊢ Eq ((CategoryTheory.StructuredArrow.map (CategoryTheory.CategoryStruct.id S) …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem map_comp {f : S ⟶ S'} {f' : S' ⟶ S''} {h : StructuredArrow S'' T} :
    (map (f ≫ f')).obj h = (map f).obj ((map f').obj h) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S S' S'' : D
    T : CategoryTheory.Functor C D
    f : Quiver.Hom S S'
    f' : Quiver.Hom S' S''
    h : CategoryTheory.StructuredArrow S'' T
    ⊢ Eq ((CategoryTheory.StructuredArrow.map (CategoryTheory.CategoryStruct.comp  …
  -/
  rw [eq_mk h]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S S' S'' : D
    T : CategoryTheory.Functor C D
    f : Quiver.Hom S S'
    f' : Quiver.Hom S' S''
    h : CategoryTheory.StructuredArrow S'' T
    ⊢ Eq ((CategoryTheory.StructuredArrow.map (CategoryTheory.CategoryStruct.comp  …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- An isomorphism `S ≅ S'` induces an equivalence `StructuredArrow S T ≌ StructuredArrow S' T`. -/
@[simp]
def mapIso (i : S ≅ S') : StructuredArrow S T ≌ StructuredArrow S' T :=
  Comma.mapLeftIso _ ((Functor.const _).mapIso i)


/-- A natural isomorphism `T ≅ T'` induces an equivalence
    `StructuredArrow S T ≌ StructuredArrow S T'`. -/
@[simp]
def mapNatIso (i : T ≅ T') : StructuredArrow S T ≌ StructuredArrow S T' :=
  Comma.mapRightIso _ i


instance proj_reflectsIsomorphisms : (proj S T).ReflectsIsomorphisms where
  reflects {Y Z} f t :=
    ⟨⟨StructuredArrow.homMk
        (inv ((proj S T).map f))
            /-
              C : Type u₁
              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝ : CategoryTheory.Category.{v₂, u₂} D
              S S' S'' : D
              Y✝ Y' Y'' : C
              T T' : CategoryTheory.Functor C D
              Y Z : CategoryTheory.StructuredArrow S T
              f : Quiver.Hom Y Z
              t : CategoryTheory.IsIso ((CategoryTheory.StructuredArrow.proj S T).map f)
              ⊢ Eq (CategoryTheory.CategoryStruct.comp Z.hom (T.map (CategoryTheory.inv ((Ca …
            -/
        (by rw [Functor.map_inv, IsIso.comp_inv_eq]; simp),
                                                     /-
                                                       🎉 no goals
                                                     -/
         /-
           C : Type u₁
           inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
           D : Type u₂
           inst✝ : CategoryTheory.Category.{v₂, u₂} D
           S S' S'' : D
           Y✝ Y' Y'' : C
           T T' : CategoryTheory.Functor C D
           Y Z : CategoryTheory.StructuredArrow S T
           f : Quiver.Hom Y Z
           t : CategoryTheory.IsIso ((CategoryTheory.StructuredArrow.proj S T).map f)
           ⊢ And (Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.StructuredArro …
         -/
                                                     /-
                                                       🎉 no goals
                                                     -/
                                                     /-
                                                       🎉 no goals
                                                     -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
      by constructor <;> apply CommaMorphism.ext <;> dsimp at t ⊢ <;> simp⟩⟩
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- The identity structured arrow is initial. -/
noncomputable def mkIdInitial [T.Full] [T.Faithful] : IsInitial (mk (𝟙 (T.obj Y))) where
            /-
              C : Type u₁
              inst✝³ : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝² : CategoryTheory.Category.{v₂, u₂} D
              S S' S'' : D
              Y Y' Y'' : C
              T T' : CategoryTheory.Functor C D
              inst✝¹ : T.Full
              inst✝ : T.Faithful
              c : CategoryTheory.Limits.Cocone (CategoryTheory.Functor.empty (CategoryTheory …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.asEmptyCocone  …
            -/
  desc c := homMk (T.preimage c.pt.hom)
            /-
              🎉 no goals
            -/
  uniq c m _ := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      S S' S'' : D
      Y Y' Y'' : C
      T T' : CategoryTheory.Functor C D
      inst✝¹ : T.Full
      inst✝ : T.Faithful
      c : CategoryTheory.Limits.Cocone (CategoryTheory.Functor.empty (CategoryTheory …
      m : Quiver.Hom (CategoryTheory.Limits.asEmptyCocone (CategoryTheory.Structured …
      x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategorySt …
      ⊢ Eq m ((fun c => CategoryTheory.StructuredArrow.homMk (T.preimage c.pt.hom) ⋯ …
    -/
    apply CommaMorphism.ext
      /-
        case left
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S S' S'' : D
        Y Y' Y'' : C
        T T' : CategoryTheory.Functor C D
        inst✝¹ : T.Full
        inst✝ : T.Faithful
        c : CategoryTheory.Limits.Cocone (CategoryTheory.Functor.empty (CategoryTheory …
        m : Quiver.Hom (CategoryTheory.Limits.asEmptyCocone (CategoryTheory.Structured …
        x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategorySt …
        ⊢ Eq m.left ((fun c => CategoryTheory.StructuredArrow.homMk (T.preimage c.pt.h …
      -/
    · aesop_cat
      /-
        🎉 no goals
      -/
      /-
        case right
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S S' S'' : D
        Y Y' Y'' : C
        T T' : CategoryTheory.Functor C D
        inst✝¹ : T.Full
        inst✝ : T.Faithful
        c : CategoryTheory.Limits.Cocone (CategoryTheory.Functor.empty (CategoryTheory …
        m : Quiver.Hom (CategoryTheory.Limits.asEmptyCocone (CategoryTheory.Structured …
        x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategorySt …
        ⊢ Eq m.right ((fun c => CategoryTheory.StructuredArrow.homMk (T.preimage c.pt. …
      -/
    · apply T.map_injective
      /-
        case right.a
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S S' S'' : D
        Y Y' Y'' : C
        T T' : CategoryTheory.Functor C D
        inst✝¹ : T.Full
        inst✝ : T.Faithful
        c : CategoryTheory.Limits.Cocone (CategoryTheory.Functor.empty (CategoryTheory …
        m : Quiver.Hom (CategoryTheory.Limits.asEmptyCocone (CategoryTheory.Structured …
        x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategorySt …
        ⊢ Eq (T.map m.right) (T.map ((fun c => CategoryTheory.StructuredArrow.homMk (T …
      -/
      simpa only [homMk_right, T.map_preimage, ← w m] using (Category.id_comp _).symm
      /-
        🎉 no goals
      -/


/-- The functor `(S, F ⋙ G) ⥤ (S, G)`. -/
@[simps!]
def pre (S : D) (F : B ⥤ C) (G : C ⥤ D) : StructuredArrow S (F ⋙ G) ⥤ StructuredArrow S G :=
  Comma.preRight _ F G


instance (S : D) (F : B ⥤ C) (G : C ⥤ D) [F.Faithful] : (pre S F G).Faithful :=
  show (Comma.preRight _ _ _).Faithful from inferInstance


instance (S : D) (F : B ⥤ C) (G : C ⥤ D) [F.Full] : (pre S F G).Full :=
  show (Comma.preRight _ _ _).Full from inferInstance


instance (S : D) (F : B ⥤ C) (G : C ⥤ D) [F.EssSurj] : (pre S F G).EssSurj :=
  show (Comma.preRight _ _ _).EssSurj from inferInstance


/-- If `F` is an equivalence, then so is the functor `(S, F ⋙ G) ⥤ (S, G)`. -/
instance isEquivalence_pre (S : D) (F : B ⥤ C) (G : C ⥤ D) [F.IsEquivalence] :
    (pre S F G).IsEquivalence :=
  Comma.isEquivalence_preRight _ _ _


/-- The functor `(S, F) ⥤ (G(S), F ⋙ G)`. -/
@[simps]
def post (S : C) (F : B ⥤ C) (G : C ⥤ D) :
    StructuredArrow S F ⥤ StructuredArrow (G.obj S) (F ⋙ G) where
  obj X := StructuredArrow.mk (G.map X.hom)
                                             /-
                                               C : Type u₁
                                               inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                               D : Type u₂
                                               inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                               S✝ S' S'' : D
                                               Y Y' Y'' : C
                                               T T' : CategoryTheory.Functor C D
                                               A : Type u₃
                                               inst✝¹ : CategoryTheory.Category.{v₃, u₃} A
                                               B : Type u₄
                                               inst✝ : CategoryTheory.Category.{v₄, u₄} B
                                               S : C
                                               F : CategoryTheory.Functor B C
                                               G : CategoryTheory.Functor C D
                                               X✝ Y✝ : CategoryTheory.StructuredArrow S F
                                               f : Quiver.Hom X✝ Y✝
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X => CategoryTheory.StructuredA …
                                             -/
  map f := StructuredArrow.homMk f.right (by simp [Functor.comp_map, ← G.map_comp, ← f.w])
                                             /-
                                               🎉 no goals
                                             -/


instance (S : C) (F : B ⥤ C) (G : C ⥤ D) : (post S F G).Faithful where
                                  /-
                                    C : Type u₁
                                    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                    D : Type u₂
                                    inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                    S✝ S' S'' : D
                                    Y Y' Y'' : C
                                    T T' : CategoryTheory.Functor C D
                                    A : Type u₃
                                    inst✝¹ : CategoryTheory.Category.{v₃, u₃} A
                                    B : Type u₄
                                    inst✝ : CategoryTheory.Category.{v₄, u₄} B
                                    S : C
                                    F : CategoryTheory.Functor B C
                                    G : CategoryTheory.Functor C D
                                    x✝³ x✝² : CategoryTheory.StructuredArrow S F
                                    x✝¹ x✝ : Quiver.Hom x✝³ x✝²
                                    h : Eq ((CategoryTheory.StructuredArrow.post S F G).map x✝¹) ((CategoryTheory. …
                                    ⊢ Eq x✝¹ x✝
                                  -/
  map_injective {_ _} _ _ h := by simpa [ext_iff] using h
                                  /-
                                    🎉 no goals
                                  -/


instance (S : C) (F : B ⥤ C) (G : C ⥤ D) [G.Faithful] : (post S F G).Full where
                                                          /-
                                                            C : Type u₁
                                                            inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                            D : Type u₂
                                                            inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                                                            S✝ S' S'' : D
                                                            Y Y' Y'' : C
                                                            T T' : CategoryTheory.Functor C D
                                                            A : Type u₃
                                                            inst✝² : CategoryTheory.Category.{v₃, u₃} A
                                                            B : Type u₄
                                                            inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
                                                            S : C
                                                            F : CategoryTheory.Functor B C
                                                            G : CategoryTheory.Functor C D
                                                            inst✝ : G.Faithful
                                                            X✝ Y✝ : CategoryTheory.StructuredArrow S F
                                                            f : Quiver.Hom ((CategoryTheory.StructuredArrow.post S F G).obj X✝) ((Category …
                                                            ⊢ Eq (G.map (CategoryTheory.CategoryStruct.comp X✝.hom (F.map f.right))) (G.ma …
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  map_surjective f := ⟨homMk f.right (G.map_injective (by simpa using f.w.symm)), by aesop_cat⟩
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


instance (S : C) (F : B ⥤ C) (G : C ⥤ D) [G.Full] : (post S F G).EssSurj where
                                                                    /-
                                                                      C : Type u₁
                                                                      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                                      D : Type u₂
                                                                      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                                                                      S✝ S' S'' : D
                                                                      Y Y' Y'' : C
                                                                      T T' : CategoryTheory.Functor C D
                                                                      A : Type u₃
                                                                      inst✝² : CategoryTheory.Category.{v₃, u₃} A
                                                                      B : Type u₄
                                                                      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
                                                                      S : C
                                                                      F : CategoryTheory.Functor B C
                                                                      G : CategoryTheory.Functor C D
                                                                      inst✝ : G.Full
                                                                      h : CategoryTheory.StructuredArrow (G.obj S) (F.comp G)
                                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.StructuredArrow.post …
                                                                    -/
  mem_essImage h := ⟨mk (G.preimage h.hom), ⟨isoMk (Iso.refl _) (by simp)⟩⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- If `G` is fully faithful, then `post S F G : (S, F) ⥤ (G(S), F ⋙ G)` is an equivalence. -/
instance isEquivalence_post (S : C) (F : B ⥤ C) (G : C ⥤ D) [G.Full] [G.Faithful] :
    (post S F G).IsEquivalence where


/-- The functor `StructuredArrow L R ⥤ StructuredArrow L' R'` that is deduced from
a natural transformation `R ⋙ G ⟶ F ⋙ R'` and a morphism `L' ⟶ G.obj L.` -/
@[simps!]
def map₂ : StructuredArrow L R ⥤ StructuredArrow L' R' :=
  Comma.map (F₁ := 𝟭 (Discrete PUnit)) (Discrete.natTrans (fun _ => α)) β


instance faithful_map₂ [F.Faithful] : (map₂ α β).Faithful := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    S S' S'' : D
    Y Y' Y'' : C
    T T' : CategoryTheory.Functor C D
    A : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} A
    B : Type u₄
    inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
    L : D
    R : CategoryTheory.Functor C D
    L' : B
    R' : CategoryTheory.Functor A B
    F : CategoryTheory.Functor C A
    G : CategoryTheory.Functor D B
    α : Quiver.Hom L' (G.obj L)
    β : Quiver.Hom (R.comp G) (F.comp R')
    inst✝ : F.Faithful
    ⊢ (CategoryTheory.StructuredArrow.map₂ α β).Faithful
  -/
  apply Comma.faithful_map
  /-
    🎉 no goals
  -/


instance full_map₂ [G.Faithful] [F.Full] [IsIso α] [IsIso β] : (map₂ α β).Full := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
    S S' S'' : D
    Y Y' Y'' : C
    T T' : CategoryTheory.Functor C D
    A : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₃, u₃} A
    B : Type u₄
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} B
    L : D
    R : CategoryTheory.Functor C D
    L' : B
    R' : CategoryTheory.Functor A B
    F : CategoryTheory.Functor C A
    G : CategoryTheory.Functor D B
    α : Quiver.Hom L' (G.obj L)
    β : Quiver.Hom (R.comp G) (F.comp R')
    inst✝³ : G.Faithful
    inst✝² : F.Full
    inst✝¹ : CategoryTheory.IsIso α
    inst✝ : CategoryTheory.IsIso β
    ⊢ (CategoryTheory.StructuredArrow.map₂ α β).Full
  -/
  apply Comma.full_map
  /-
    🎉 no goals
  -/


instance essSurj_map₂ [F.EssSurj] [G.Full] [IsIso α] [IsIso β] : (map₂ α β).EssSurj := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
    S S' S'' : D
    Y Y' Y'' : C
    T T' : CategoryTheory.Functor C D
    A : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₃, u₃} A
    B : Type u₄
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} B
    L : D
    R : CategoryTheory.Functor C D
    L' : B
    R' : CategoryTheory.Functor A B
    F : CategoryTheory.Functor C A
    G : CategoryTheory.Functor D B
    α : Quiver.Hom L' (G.obj L)
    β : Quiver.Hom (R.comp G) (F.comp R')
    inst✝³ : F.EssSurj
    inst✝² : G.Full
    inst✝¹ : CategoryTheory.IsIso α
    inst✝ : CategoryTheory.IsIso β
    ⊢ (CategoryTheory.StructuredArrow.map₂ α β).EssSurj
  -/
  apply Comma.essSurj_map
  /-
    🎉 no goals
  -/


noncomputable instance isEquivalenceMap₂
    [F.IsEquivalence] [G.Faithful] [G.Full] [IsIso α] [IsIso β] :
    (map₂ α β).IsEquivalence := by
  /-
    C : Type u₁
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
    S S' S'' : D
    Y Y' Y'' : C
    T T' : CategoryTheory.Functor C D
    A : Type u₃
    inst✝⁶ : CategoryTheory.Category.{v₃, u₃} A
    B : Type u₄
    inst✝⁵ : CategoryTheory.Category.{v₄, u₄} B
    L : D
    R : CategoryTheory.Functor C D
    L' : B
    R' : CategoryTheory.Functor A B
    F : CategoryTheory.Functor C A
    G : CategoryTheory.Functor D B
    α : Quiver.Hom L' (G.obj L)
    β : Quiver.Hom (R.comp G) (F.comp R')
    inst✝⁴ : F.IsEquivalence
    inst✝³ : G.Faithful
    inst✝² : G.Full
    inst✝¹ : CategoryTheory.IsIso α
    inst✝ : CategoryTheory.IsIso β
    ⊢ (CategoryTheory.StructuredArrow.map₂ α β).IsEquivalence
  -/
  apply Comma.isEquivalenceMap
  /-
    🎉 no goals
  -/


/-- `StructuredArrow.post` is a special case of `StructuredArrow.map₂` up to natural isomorphism. -/
def postIsoMap₂ (S : C) (F : B ⥤ C) (G : C ⥤ D) :
    post S F G ≅ map₂ (F := 𝟭 _) (𝟙 _) (𝟙 (F ⋙ G)) :=
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    S✝ S' S'' : D
    Y Y' Y'' : C
    T T' : CategoryTheory.Functor C D
    A : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} A
    B : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} B
    S : C
    F : CategoryTheory.Functor B C
    G : CategoryTheory.Functor C D
    ⊢ ∀ {X Y : CategoryTheory.StructuredArrow S F} (f : Quiver.Hom X Y), Eq (Categ …
  -/
  NatIso.ofComponents fun _ => isoMk <| Iso.refl _
  /-
    🎉 no goals
  -/


/-- A structured arrow is called universal if it is initial. -/
abbrev IsUniversal (f : StructuredArrow S T) := IsInitial f


theorem uniq (h : IsUniversal f) (η : f ⟶ g) : η = h.to g :=
  h.hom_ext η (h.to g)


/-- The family of morphisms out of a universal arrow. -/
def desc (h : IsUniversal f) (g : StructuredArrow S T) : f.right ⟶ g.right :=
  (h.to g).right


/-- Any structured arrow factors through a universal arrow. -/
@[reassoc (attr := simp)]
theorem fac (h : IsUniversal f) (g : StructuredArrow S T) :
    f.hom ≫ T.map (h.desc g) = g.hom :=
  Category.id_comp g.hom ▸ (h.to g).w.symm


theorem hom_desc (h : IsUniversal f) {c : C} (η : f.right ⟶ c) :
    η = h.desc (mk <| f.hom ≫ T.map η) :=
  let g := mk <| f.hom ≫ T.map η
  congrArg CommaMorphism.right (h.hom_ext (homMk η rfl : f ⟶ g) (h.to g))


/-- Two morphisms out of a universal `T`-structured arrow are equal if their image under `T` are
equal after precomposing the universal arrow. -/
theorem hom_ext (h : IsUniversal f) {c : C} {η η' : f.right ⟶ c}
    (w : f.hom ≫ T.map η = f.hom ≫ T.map η') : η = η' := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    S : D
    T : CategoryTheory.Functor C D
    f : CategoryTheory.StructuredArrow S T
    h : f.IsUniversal
    c : C
    η η' : Quiver.Hom f.right c
    w : Eq (CategoryTheory.CategoryStruct.comp f.hom (T.map η)) (CategoryTheory.Ca …
    ⊢ Eq η η'
  -/
  rw [h.hom_desc η, h.hom_desc η', w]
  /-
    🎉 no goals
  -/


theorem existsUnique (h : IsUniversal f) (g : StructuredArrow S T) :
    ∃! η : f.right ⟶ g.right, f.hom ≫ T.map η = g.hom :=
                                                /-
                                                  C : Type u₁
                                                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                  D : Type u₂
                                                  inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                  S : D
                                                  T : CategoryTheory.Functor C D
                                                  f✝ : CategoryTheory.StructuredArrow S T
                                                  h : f✝.IsUniversal
                                                  g : CategoryTheory.StructuredArrow S T
                                                  f : Quiver.Hom f✝.right g.right
                                                  w : (fun η => Eq (CategoryTheory.CategoryStruct.comp f✝.hom (T.map η)) g.hom) f
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp f✝.hom (T.map f)) (CategoryTheory.Cat …
                                                -/
  ⟨h.desc g, h.fac g, fun f w ↦ h.hom_ext <| by simp [w]⟩
                                                /-
                                                  🎉 no goals
                                                -/


/-- The category of `S`-costructured arrows with target `T : D` (here `S : C ⥤ D`),
has as its objects `D`-morphisms of the form `S Y ⟶ T`, for some `Y : C`,
and morphisms `C`-morphisms `Y ⟶ Y'` making the obvious triangle commute.
-/
-- We explicitly come from `PUnit.{1}` here to obtain the correct universe for morphisms of
-- costructured arrows.
-- @[nolint has_nonempty_instance] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): linter not ported yet
def CostructuredArrow (S : C ⥤ D) (T : D) :=
  Comma S (Functor.fromPUnit.{0} T)


instance (S : C ⥤ D) (T : D) : Category (CostructuredArrow S T) := commaCategory


/-- The obvious projection functor from costructured arrows. -/
@[simps!]
def proj (S : C ⥤ D) (T : D) : CostructuredArrow S T ⥤ C :=
  Comma.fst _ _


@[ext]
lemma hom_ext {X Y : CostructuredArrow S T} (f g : X ⟶ Y) (h : f.left = g.left) : f = g :=
  CommaMorphism.ext h (Subsingleton.elim _ _)


@[simp]
theorem hom_eq_iff {X Y : CostructuredArrow S T} (f g : X ⟶ Y) : f = g ↔ f.left = g.left :=
              /-
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                D : Type u₂
                inst✝ : CategoryTheory.Category.{v₂, u₂} D
                T : D
                S : CategoryTheory.Functor C D
                X Y : CategoryTheory.CostructuredArrow S T
                f g : Quiver.Hom X Y
                h : Eq f g
                ⊢ Eq f.left g.left
              -/
  ⟨fun h ↦ by rw [h], hom_ext _ _⟩
              /-
                🎉 no goals
              -/


/-- Construct a costructured arrow from a morphism. -/
def mk (f : S.obj Y ⟶ T) : CostructuredArrow S T :=
  ⟨Y, ⟨⟨⟩⟩, f⟩


@[simp]
theorem mk_left (f : S.obj Y ⟶ T) : (mk f).left = Y :=
  rfl


@[simp]
theorem mk_right (f : S.obj Y ⟶ T) : (mk f).right = ⟨⟨⟩⟩ :=
  rfl


@[simp]
theorem mk_hom_eq_self (f : S.obj Y ⟶ T) : (mk f).hom = f :=
  rfl

-- @[reassoc (attr := simp)] Porting note: simp can solve these

@[reassoc]
                                                                                         /-
                                                                                           C : Type u₁
                                                                                           inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                           D : Type u₂
                                                                                           inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                                           T : D
                                                                                           S : CategoryTheory.Functor C D
                                                                                           A B : CategoryTheory.CostructuredArrow S T
                                                                                           f : Quiver.Hom A B
                                                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map f.left) B.hom) A.hom
                                                                                         -/
theorem w {A B : CostructuredArrow S T} (f : A ⟶ B) : S.map f.left ≫ B.hom = A.hom := by simp
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


@[simp]
theorem comp_left {X Y Z : CostructuredArrow S T} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).left = f.left ≫ g.left := rfl


@[simp]
theorem id_left (X : CostructuredArrow S T) : (𝟙 X : X ⟶ X).left = 𝟙 X.left := rfl


@[simp]
theorem eqToHom_left {X Y : CostructuredArrow S T} (h : X = Y) :
                                   /-
                                     C : Type u₁
                                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                     D : Type u₂
                                     inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                     T T' T'' : D
                                     Y✝ Y' Y'' : C
                                     S S' : CategoryTheory.Functor C D
                                     X Y : CategoryTheory.CostructuredArrow S T
                                     h : Eq X Y
                                     ⊢ Eq X.left Y.left
                                   -/
    (eqToHom h).left = eqToHom (by rw [h]) := by
                                   /-
                                     🎉 no goals
                                   -/
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    T : D
    S : CategoryTheory.Functor C D
    X Y : CategoryTheory.CostructuredArrow S T
    h : Eq X Y
    ⊢ Eq (CategoryTheory.eqToHom h).left (CategoryTheory.eqToHom ⋯)
  -/
  subst h
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    T : D
    S : CategoryTheory.Functor C D
    X : CategoryTheory.CostructuredArrow S T
    ⊢ Eq (CategoryTheory.eqToHom ⋯).left (CategoryTheory.eqToHom ⋯)
  -/
  simp only [eqToHom_refl, id_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem right_eq_id {X Y : CostructuredArrow S T} (f : X ⟶ Y) : f.right = 𝟙 X.right := rfl


/-- To construct a morphism of costructured arrows,
we need a morphism of the objects underlying the source,
and to check that the triangle commutes.
-/
@[simps!]
def homMk {f f' : CostructuredArrow S T} (g : f.left ⟶ f'.left)
    (w : S.map g ≫ f'.hom = f.hom := by aesop_cat) : f ⟶ f' where
  left := g
  right := 𝟙 f.right

/- Porting note: it appears the simp lemma is not getting generated but the linter
picks up on it. Either way simp can prove this -/

theorem homMk_surjective {f f' : CostructuredArrow S T} (φ : f ⟶ f') :
    ∃ (ψ : f.left ⟶ f'.left) (hψ : S.map ψ ≫ f'.hom = f.hom),
      φ = CostructuredArrow.homMk ψ hψ :=
  ⟨φ.left, CostructuredArrow.w φ, rfl⟩


/-- Given a costructured arrow `S(Y) ⟶ X`, and an arrow `Y' ⟶ Y'`, we can construct a morphism of
    costructured arrows given by `(S(Y) ⟶ X) ⟶ (S(Y') ⟶ S(Y) ⟶ X)`. -/
@[simps]
def homMk' (f : CostructuredArrow S T) (g : Y' ⟶ f.left) : mk (S.map g ≫ f.hom) ⟶ f where
  left := g
  right := 𝟙 _


                                                                                /-
                                                                                  C : Type u₁
                                                                                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                  D : Type u₂
                                                                                  inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                                  T T' T'' : D
                                                                                  Y Y' Y'' : C
                                                                                  S S' : CategoryTheory.Functor C D
                                                                                  f : CategoryTheory.CostructuredArrow S T
                                                                                  ⊢ Eq (CategoryTheory.CostructuredArrow.mk (CategoryTheory.CategoryStruct.comp  …
                                                                                -/
lemma homMk'_id (f : CostructuredArrow S T) : homMk' f (𝟙 f.left) = eqToHom (by aesop_cat) := by
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    T : D
    S : CategoryTheory.Functor C D
    f : CategoryTheory.CostructuredArrow S T
    ⊢ Eq (f.homMk' (CategoryTheory.CategoryStruct.id f.left)) (CategoryTheory.eqTo …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    T : D
    S : CategoryTheory.Functor C D
    f : CategoryTheory.CostructuredArrow S T
    ⊢ Eq (f.homMk' (CategoryTheory.CategoryStruct.id f.left)).left (CategoryTheory …
  -/
  simp [eqToHom_left]
  /-
    🎉 no goals
  -/


                                                                         /-
                                                                           C : Type u₁
                                                                           inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                           D : Type u₂
                                                                           inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                           T T' T'' : D
                                                                           Y Y' Y'' : C
                                                                           S S' : CategoryTheory.Functor C D
                                                                           f : Quiver.Hom (S.obj Y) T
                                                                           ⊢ Eq (CategoryTheory.CostructuredArrow.mk (CategoryTheory.CategoryStruct.comp  …
                                                                         -/
lemma homMk'_mk_id (f : S.obj Y ⟶ T) : homMk' (mk f) (𝟙 Y) = eqToHom (by aesop_cat) :=
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
  homMk'_id _


lemma homMk'_comp (f : CostructuredArrow S T) (g : Y' ⟶ f.left) (g' : Y'' ⟶ Y') :
                                    /-
                                      C : Type u₁
                                      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                      D : Type u₂
                                      inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                      T T' T'' : D
                                      Y Y' Y'' : C
                                      S S' : CategoryTheory.Functor C D
                                      f : CategoryTheory.CostructuredArrow S T
                                      g : Quiver.Hom Y' f.left
                                      g' : Quiver.Hom Y'' Y'
                                      ⊢ Eq (CategoryTheory.CostructuredArrow.mk (CategoryTheory.CategoryStruct.comp  …
                                    -/
    homMk' f (g' ≫ g) = eqToHom (by simp) ≫ homMk' (mk (S.map g ≫ f.hom)) g' ≫ homMk' f g := by
                                    /-
                                      🎉 no goals
                                    -/
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    T : D
    Y' Y'' : C
    S : CategoryTheory.Functor C D
    f : CategoryTheory.CostructuredArrow S T
    g : Quiver.Hom Y' f.left
    g' : Quiver.Hom Y'' Y'
    ⊢ Eq (f.homMk' (CategoryTheory.CategoryStruct.comp g' g)) (CategoryTheory.Cate …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    T : D
    Y' Y'' : C
    S : CategoryTheory.Functor C D
    f : CategoryTheory.CostructuredArrow S T
    g : Quiver.Hom Y' f.left
    g' : Quiver.Hom Y'' Y'
    ⊢ Eq (f.homMk' (CategoryTheory.CategoryStruct.comp g' g)).left (CategoryTheory …
  -/
  simp [eqToHom_left]
  /-
    🎉 no goals
  -/


lemma homMk'_mk_comp (f : S.obj Y ⟶ T) (g : Y' ⟶ Y) (g' : Y'' ⟶ Y') :
                                         /-
                                           C : Type u₁
                                           inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                           D : Type u₂
                                           inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                           T T' T'' : D
                                           Y Y' Y'' : C
                                           S S' : CategoryTheory.Functor C D
                                           f : Quiver.Hom (S.obj Y) T
                                           g : Quiver.Hom Y' Y
                                           g' : Quiver.Hom Y'' Y'
                                           ⊢ Eq (CategoryTheory.CostructuredArrow.mk (CategoryTheory.CategoryStruct.comp  …
                                         -/
    homMk' (mk f) (g' ≫ g) = eqToHom (by simp) ≫ homMk' (mk (S.map g ≫ f)) g' ≫ homMk' (mk f) g :=
                                         /-
                                           🎉 no goals
                                         -/
  homMk'_comp _ _ _


/-- Variant of `homMk'` where both objects are applications of `mk`. -/
@[simps]
def mkPrecomp (f : S.obj Y ⟶ T) (g : Y' ⟶ Y) : mk (S.map g ≫ f) ⟶ mk f where
  left := g
  right := 𝟙 _


                                                                       /-
                                                                         C : Type u₁
                                                                         inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                         D : Type u₂
                                                                         inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                         T T' T'' : D
                                                                         Y Y' Y'' : C
                                                                         S S' : CategoryTheory.Functor C D
                                                                         f : Quiver.Hom (S.obj Y) T
                                                                         ⊢ Eq (CategoryTheory.CostructuredArrow.mk (CategoryTheory.CategoryStruct.comp  …
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
lemma mkPrecomp_id (f : S.obj Y ⟶ T) : mkPrecomp f (𝟙 Y) = eqToHom (by aesop_cat) := by aesop_cat
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/

lemma mkPrecomp_comp (f : S.obj Y ⟶ T) (g : Y' ⟶ Y) (g' : Y'' ⟶ Y') :
                                       /-
                                         C : Type u₁
                                         inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                         D : Type u₂
                                         inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                         T T' T'' : D
                                         Y Y' Y'' : C
                                         S S' : CategoryTheory.Functor C D
                                         f : Quiver.Hom (S.obj Y) T
                                         g : Quiver.Hom Y' Y
                                         g' : Quiver.Hom Y'' Y'
                                         ⊢ Eq (CategoryTheory.CostructuredArrow.mk (CategoryTheory.CategoryStruct.comp  …
                                       -/
    mkPrecomp f (g' ≫ g) = eqToHom (by simp) ≫ mkPrecomp (S.map g ≫ f) g' ≫ mkPrecomp f g := by
                                       /-
                                         🎉 no goals
                                       -/
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    T : D
    Y Y' Y'' : C
    S : CategoryTheory.Functor C D
    f : Quiver.Hom (S.obj Y) T
    g : Quiver.Hom Y' Y
    g' : Quiver.Hom Y'' Y'
    ⊢ Eq (CategoryTheory.CostructuredArrow.mkPrecomp f (CategoryTheory.CategoryStr …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


/-- To construct an isomorphism of costructured arrows,
we need an isomorphism of the objects underlying the source,
and to check that the triangle commutes.
-/
@[simps!]
def isoMk {f f' : CostructuredArrow S T} (g : f.left ≅ f'.left)
    (w : S.map g.hom ≫ f'.hom = f.hom := by aesop_cat) : f ≅ f' :=
                             /-
                               C : Type u₁
                               inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                               D : Type u₂
                               inst✝ : CategoryTheory.Category.{v₂, u₂} D
                               T T' T'' : D
                               Y Y' Y'' : C
                               S S' : CategoryTheory.Functor C D
                               f f' : CategoryTheory.CostructuredArrow S T
                               g : CategoryTheory.Iso f.left f'.left
                               w : autoParam (Eq (CategoryTheory.CategoryStruct.comp (S.map g.hom) f'.hom) f. …
                               ⊢ Eq f.right f'.right
                             -/
                             /-
                               🎉 no goals
                             -/
  Comma.isoMk g (eqToIso (by ext)) (by simpa using w)
                                       /-
                                         🎉 no goals
                                       -/

/- Porting note: it appears the simp lemma is not getting generated but the linter
picks up on it. Either way simp solves these. -/

theorem ext {A B : CostructuredArrow S T} (f g : A ⟶ B) (h : f.left = g.left) : f = g :=
  CommaMorphism.ext h (Subsingleton.elim _ _)


theorem ext_iff {A B : CostructuredArrow S T} (f g : A ⟶ B) : f = g ↔ f.left = g.left :=
  ⟨fun h => h ▸ rfl, ext f g⟩


instance proj_faithful : (proj S T).Faithful where map_injective {_ _} := ext


theorem mono_of_mono_left {A B : CostructuredArrow S T} (f : A ⟶ B) [h : Mono f.left] : Mono f :=
  (proj S T).mono_of_mono_map h


/-- The converse of this is true with additional assumptions, see `epi_iff_epi_left`. -/
theorem epi_of_epi_left {A B : CostructuredArrow S T} (f : A ⟶ B) [h : Epi f.left] : Epi f :=
  (proj S T).epi_of_epi_map h


instance mono_homMk {A B : CostructuredArrow S T} (f : A.left ⟶ B.left) (w) [h : Mono f] :
    Mono (homMk f w) :=
  (proj S T).mono_of_mono_map h


instance epi_homMk {A B : CostructuredArrow S T} (f : A.left ⟶ B.left) (w) [h : Epi f] :
    Epi (homMk f w) :=
  (proj S T).epi_of_epi_map h


/-- Eta rule for costructured arrows. Prefer `CostructuredArrow.eta` for rewriting, as equality of
    objects tends to cause problems. -/
theorem eq_mk (f : CostructuredArrow S T) : f = mk f.hom :=
  rfl


/-- Eta rule for costructured arrows. -/
@[simps!]
def eta (f : CostructuredArrow S T) : f ≅ mk f.hom :=
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    T T' T'' : D
    Y Y' Y'' : C
    S S' : CategoryTheory.Functor C D
    f : CategoryTheory.CostructuredArrow S T
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map (CategoryTheory.Iso.refl f.lef …
  -/
  isoMk (Iso.refl _)
  /-
    🎉 no goals
  -/

/- Porting note: it appears the simp lemma is not getting generated but the linter
picks up on it. Either way simp solves these. -/

lemma mk_surjective (f : CostructuredArrow S T) :
    ∃ (Y : C) (g : S.obj Y ⟶ T), f = mk g :=
  ⟨_, _, eq_mk f⟩


/-- A morphism between target objects `T ⟶ T'`
covariantly induces a functor between costructured arrows,
`CostructuredArrow S T ⥤ CostructuredArrow S T'`.

Ideally this would be described as a 2-functor from `D`
(promoted to a 2-category with equations as 2-morphisms)
to `Cat`.
-/
@[simps!]
def map (f : T ⟶ T') : CostructuredArrow S T ⥤ CostructuredArrow S T' :=
  Comma.mapRight _ ((Functor.const _).map f)


@[simp]
theorem map_mk {f : S.obj Y ⟶ T} (g : T ⟶ T') : (map g).obj (mk f) = mk (f ≫ g) :=
  rfl


@[simp]
theorem map_id {f : CostructuredArrow S T} : (map (𝟙 T)).obj f = f := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    T : D
    S : CategoryTheory.Functor C D
    f : CategoryTheory.CostructuredArrow S T
    ⊢ Eq ((CategoryTheory.CostructuredArrow.map (CategoryTheory.CategoryStruct.id  …
  -/
  rw [eq_mk f]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    T : D
    S : CategoryTheory.Functor C D
    f : CategoryTheory.CostructuredArrow S T
    ⊢ Eq ((CategoryTheory.CostructuredArrow.map (CategoryTheory.CategoryStruct.id  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem map_comp {f : T ⟶ T'} {f' : T' ⟶ T''} {h : CostructuredArrow S T} :
    (map (f ≫ f')).obj h = (map f').obj ((map f).obj h) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    T T' T'' : D
    S : CategoryTheory.Functor C D
    f : Quiver.Hom T T'
    f' : Quiver.Hom T' T''
    h : CategoryTheory.CostructuredArrow S T
    ⊢ Eq ((CategoryTheory.CostructuredArrow.map (CategoryTheory.CategoryStruct.com …
  -/
  rw [eq_mk h]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    T T' T'' : D
    S : CategoryTheory.Functor C D
    f : Quiver.Hom T T'
    f' : Quiver.Hom T' T''
    h : CategoryTheory.CostructuredArrow S T
    ⊢ Eq ((CategoryTheory.CostructuredArrow.map (CategoryTheory.CategoryStruct.com …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- An isomorphism `T ≅ T'` induces an equivalence
    `CostructuredArrow S T ≌ CostructuredArrow S T'`. -/
@[simp]
def mapIso (i : T ≅ T') : CostructuredArrow S T ≌ CostructuredArrow S T' :=
  Comma.mapRightIso _ ((Functor.const _).mapIso i)


/-- A natural isomorphism `S ≅ S'` induces an equivalence
    `CostrucutredArrow S T ≌ CostructuredArrow S' T`. -/
@[simp]
def mapNatIso (i : S ≅ S') : CostructuredArrow S T ≌ CostructuredArrow S' T :=
  Comma.mapLeftIso _ i


instance proj_reflectsIsomorphisms : (proj S T).ReflectsIsomorphisms where
  reflects {Y Z} f t :=
    ⟨⟨CostructuredArrow.homMk
        (inv ((proj S T).map f))
            /-
              C : Type u₁
              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝ : CategoryTheory.Category.{v₂, u₂} D
              T T' T'' : D
              Y✝ Y' Y'' : C
              S S' : CategoryTheory.Functor C D
              Y Z : CategoryTheory.CostructuredArrow S T
              f : Quiver.Hom Y Z
              t : CategoryTheory.IsIso ((CategoryTheory.CostructuredArrow.proj S T).map f)
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map (CategoryTheory.inv ((Category …
            -/
        (by rw [Functor.map_inv, IsIso.inv_comp_eq]; simp),
                                                     /-
                                                       🎉 no goals
                                                     -/
         /-
           C : Type u₁
           inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
           D : Type u₂
           inst✝ : CategoryTheory.Category.{v₂, u₂} D
           T T' T'' : D
           Y✝ Y' Y'' : C
           S S' : CategoryTheory.Functor C D
           Y Z : CategoryTheory.CostructuredArrow S T
           f : Quiver.Hom Y Z
           t : CategoryTheory.IsIso ((CategoryTheory.CostructuredArrow.proj S T).map f)
           ⊢ And (Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CostructuredAr …
         -/
                                                  /-
                                                    🎉 no goals
                                                  -/
      by constructor <;> ext <;> dsimp at t ⊢ <;> simp⟩⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- The identity costructured arrow is terminal. -/
noncomputable def mkIdTerminal [S.Full] [S.Faithful] : IsTerminal (mk (𝟙 (S.obj Y))) where
            /-
              C : Type u₁
              inst✝³ : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝² : CategoryTheory.Category.{v₂, u₂} D
              T T' T'' : D
              Y Y' Y'' : C
              S S' : CategoryTheory.Functor C D
              inst✝¹ : S.Full
              inst✝ : S.Faithful
              c : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty (CategoryTheory.C …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map (S.preimage c.pt.hom)) (Catego …
            -/
  lift c := homMk (S.preimage c.pt.hom)
            /-
              🎉 no goals
            -/
  uniq := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      T T' T'' : D
      Y Y' Y'' : C
      S S' : CategoryTheory.Functor C D
      inst✝¹ : S.Full
      inst✝ : S.Faithful
      ⊢ ∀ (s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty (CategoryThe …
    -/
    rintro c m -
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      T T' T'' : D
      Y Y' Y'' : C
      S S' : CategoryTheory.Functor C D
      inst✝¹ : S.Full
      inst✝ : S.Faithful
      c : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty (CategoryTheory.C …
      m : Quiver.Hom c.pt (CategoryTheory.Limits.asEmptyCone (CategoryTheory.Costruc …
      ⊢ Eq m ((fun c => CategoryTheory.CostructuredArrow.homMk (S.preimage c.pt.hom) …
    -/
    ext
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      T T' T'' : D
      Y Y' Y'' : C
      S S' : CategoryTheory.Functor C D
      inst✝¹ : S.Full
      inst✝ : S.Faithful
      c : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty (CategoryTheory.C …
      m : Quiver.Hom c.pt (CategoryTheory.Limits.asEmptyCone (CategoryTheory.Costruc …
      ⊢ Eq m.left ((fun c => CategoryTheory.CostructuredArrow.homMk (S.preimage c.pt …
    -/
    apply S.map_injective
    /-
      case h.a
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      T T' T'' : D
      Y Y' Y'' : C
      S S' : CategoryTheory.Functor C D
      inst✝¹ : S.Full
      inst✝ : S.Faithful
      c : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty (CategoryTheory.C …
      m : Quiver.Hom c.pt (CategoryTheory.Limits.asEmptyCone (CategoryTheory.Costruc …
      ⊢ Eq (S.map m.left) (S.map ((fun c => CategoryTheory.CostructuredArrow.homMk ( …
    -/
    simpa only [homMk_left, S.map_preimage, ← w m] using (Category.comp_id _).symm
    /-
      🎉 no goals
    -/


/-- The functor `(F ⋙ G, S) ⥤ (G, S)`. -/
@[simps!]
def pre (F : B ⥤ C) (G : C ⥤ D) (S : D) : CostructuredArrow (F ⋙ G) S ⥤ CostructuredArrow G S :=
  Comma.preLeft F G _


instance (F : B ⥤ C) (G : C ⥤ D) (S : D) [F.Faithful] : (pre F G S).Faithful :=
  show (Comma.preLeft _ _ _).Faithful from inferInstance


instance (F : B ⥤ C) (G : C ⥤ D) (S : D) [F.Full] : (pre F G S).Full :=
  show (Comma.preLeft _ _ _).Full from inferInstance


instance (F : B ⥤ C) (G : C ⥤ D) (S : D) [F.EssSurj] : (pre F G S).EssSurj :=
  show (Comma.preLeft _ _ _).EssSurj from inferInstance


/-- If `F` is an equivalence, then so is the functor `(F ⋙ G, S) ⥤ (G, S)`. -/
instance isEquivalence_pre (F : B ⥤ C) (G : C ⥤ D) (S : D) [F.IsEquivalence] :
    (pre F G S).IsEquivalence :=
  Comma.isEquivalence_preLeft _ _ _


/-- The functor `(F, S) ⥤ (F ⋙ G, G(S))`. -/
@[simps]
def post (F : B ⥤ C) (G : C ⥤ D) (S : C) :
    CostructuredArrow F S ⥤ CostructuredArrow (F ⋙ G) (G.obj S) where
  obj X := CostructuredArrow.mk (G.map X.hom)
                                              /-
                                                C : Type u₁
                                                inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                                D : Type u₂
                                                inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                T T' T'' : D
                                                Y Y' Y'' : C
                                                S✝ S' : CategoryTheory.Functor C D
                                                A : Type u₃
                                                inst✝¹ : CategoryTheory.Category.{v₃, u₃} A
                                                B : Type u₄
                                                inst✝ : CategoryTheory.Category.{v₄, u₄} B
                                                F : CategoryTheory.Functor B C
                                                G : CategoryTheory.Functor C D
                                                S : C
                                                X✝ Y✝ : CategoryTheory.CostructuredArrow F S
                                                f : Quiver.Hom X✝ Y✝
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.comp G).map f.left) ((fun X => Ca …
                                              -/
  map f := CostructuredArrow.homMk f.left (by simp [Functor.comp_map, ← G.map_comp, ← f.w])
                                              /-
                                                🎉 no goals
                                              -/


instance (F : B ⥤ C) (G : C ⥤ D) (S : C) : (post F G S).Faithful where
                                  /-
                                    C : Type u₁
                                    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                    D : Type u₂
                                    inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                    T T' T'' : D
                                    Y Y' Y'' : C
                                    S✝ S' : CategoryTheory.Functor C D
                                    A : Type u₃
                                    inst✝¹ : CategoryTheory.Category.{v₃, u₃} A
                                    B : Type u₄
                                    inst✝ : CategoryTheory.Category.{v₄, u₄} B
                                    F : CategoryTheory.Functor B C
                                    G : CategoryTheory.Functor C D
                                    S : C
                                    x✝³ x✝² : CategoryTheory.CostructuredArrow F S
                                    x✝¹ x✝ : Quiver.Hom x✝³ x✝²
                                    h : Eq ((CategoryTheory.CostructuredArrow.post F G S).map x✝¹) ((CategoryTheor …
                                    ⊢ Eq x✝¹ x✝
                                  -/
  map_injective {_ _} _ _ h := by simpa [ext_iff] using h
                                  /-
                                    🎉 no goals
                                  -/


instance (F : B ⥤ C) (G : C ⥤ D) (S : C) [G.Faithful] : (post F G S).Full where
                                                         /-
                                                           C : Type u₁
                                                           inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                           D : Type u₂
                                                           inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                                                           T T' T'' : D
                                                           Y Y' Y'' : C
                                                           S✝ S' : CategoryTheory.Functor C D
                                                           A : Type u₃
                                                           inst✝² : CategoryTheory.Category.{v₃, u₃} A
                                                           B : Type u₄
                                                           inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
                                                           F : CategoryTheory.Functor B C
                                                           G : CategoryTheory.Functor C D
                                                           S : C
                                                           inst✝ : G.Faithful
                                                           X✝ Y✝ : CategoryTheory.CostructuredArrow F S
                                                           f : Quiver.Hom ((CategoryTheory.CostructuredArrow.post F G S).obj X✝) ((Catego …
                                                           ⊢ Eq (G.map (CategoryTheory.CategoryStruct.comp (F.map f.left) Y✝.hom)) (G.map …
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
  map_surjective f := ⟨homMk f.left (G.map_injective (by simpa using f.w)), by aesop_cat⟩
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


instance (F : B ⥤ C) (G : C ⥤ D) (S : C) [G.Full] : (post F G S).EssSurj where
                                                                    /-
                                                                      C : Type u₁
                                                                      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                                      D : Type u₂
                                                                      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                                                                      T T' T'' : D
                                                                      Y Y' Y'' : C
                                                                      S✝ S' : CategoryTheory.Functor C D
                                                                      A : Type u₃
                                                                      inst✝² : CategoryTheory.Category.{v₃, u₃} A
                                                                      B : Type u₄
                                                                      inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
                                                                      F : CategoryTheory.Functor B C
                                                                      G : CategoryTheory.Functor C D
                                                                      S : C
                                                                      inst✝ : G.Full
                                                                      h : CategoryTheory.CostructuredArrow (F.comp G) (G.obj S)
                                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.comp G).map (CategoryTheory.Iso.r …
                                                                    -/
  mem_essImage h := ⟨mk (G.preimage h.hom), ⟨isoMk (Iso.refl _) (by simp)⟩⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- If `G` is fully faithful, then `post F G S : (F, S) ⥤ (F ⋙ G, G(S))` is an equivalence. -/
instance isEquivalence_post (S : C) (F : B ⥤ C) (G : C ⥤ D) [G.Full] [G.Faithful] :
    (post F G S).IsEquivalence where


/-- The functor `CostructuredArrow S T ⥤ CostructuredArrow U V` that is deduced from
a natural transformation `F ⋙ U ⟶ S ⋙ G` and a morphism `G.obj T ⟶ V` -/
@[simps!]
def map₂ : CostructuredArrow S T ⥤ CostructuredArrow U V :=
  Comma.map (F₂ := 𝟭 (Discrete PUnit)) α (Discrete.natTrans (fun _ => β))


instance faithful_map₂ [F.Faithful] : (map₂ α β).Faithful := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    T T' T'' : D
    Y Y' Y'' : C
    S S' : CategoryTheory.Functor C D
    A : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} A
    B : Type u₄
    inst✝¹ : CategoryTheory.Category.{v₄, u₄} B
    U : CategoryTheory.Functor A B
    V : B
    F : CategoryTheory.Functor C A
    G : CategoryTheory.Functor D B
    α : Quiver.Hom (F.comp U) (S.comp G)
    β : Quiver.Hom (G.obj T) V
    inst✝ : F.Faithful
    ⊢ (CategoryTheory.CostructuredArrow.map₂ α β).Faithful
  -/
  apply Comma.faithful_map
  /-
    🎉 no goals
  -/


instance full_map₂ [G.Faithful] [F.Full] [IsIso α] [IsIso β] : (map₂ α β).Full := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
    T T' T'' : D
    Y Y' Y'' : C
    S S' : CategoryTheory.Functor C D
    A : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₃, u₃} A
    B : Type u₄
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} B
    U : CategoryTheory.Functor A B
    V : B
    F : CategoryTheory.Functor C A
    G : CategoryTheory.Functor D B
    α : Quiver.Hom (F.comp U) (S.comp G)
    β : Quiver.Hom (G.obj T) V
    inst✝³ : G.Faithful
    inst✝² : F.Full
    inst✝¹ : CategoryTheory.IsIso α
    inst✝ : CategoryTheory.IsIso β
    ⊢ (CategoryTheory.CostructuredArrow.map₂ α β).Full
  -/
  apply Comma.full_map
  /-
    🎉 no goals
  -/


instance essSurj_map₂ [F.EssSurj] [G.Full] [IsIso α] [IsIso β] : (map₂ α β).EssSurj := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
    T T' T'' : D
    Y Y' Y'' : C
    S S' : CategoryTheory.Functor C D
    A : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₃, u₃} A
    B : Type u₄
    inst✝⁴ : CategoryTheory.Category.{v₄, u₄} B
    U : CategoryTheory.Functor A B
    V : B
    F : CategoryTheory.Functor C A
    G : CategoryTheory.Functor D B
    α : Quiver.Hom (F.comp U) (S.comp G)
    β : Quiver.Hom (G.obj T) V
    inst✝³ : F.EssSurj
    inst✝² : G.Full
    inst✝¹ : CategoryTheory.IsIso α
    inst✝ : CategoryTheory.IsIso β
    ⊢ (CategoryTheory.CostructuredArrow.map₂ α β).EssSurj
  -/
  apply Comma.essSurj_map
  /-
    🎉 no goals
  -/


noncomputable instance isEquivalenceMap₂
    [F.IsEquivalence] [G.Faithful] [G.Full] [IsIso α] [IsIso β] :
    (map₂ α β).IsEquivalence := by
  /-
    C : Type u₁
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁷ : CategoryTheory.Category.{v₂, u₂} D
    T T' T'' : D
    Y Y' Y'' : C
    S S' : CategoryTheory.Functor C D
    A : Type u₃
    inst✝⁶ : CategoryTheory.Category.{v₃, u₃} A
    B : Type u₄
    inst✝⁵ : CategoryTheory.Category.{v₄, u₄} B
    U : CategoryTheory.Functor A B
    V : B
    F : CategoryTheory.Functor C A
    G : CategoryTheory.Functor D B
    α : Quiver.Hom (F.comp U) (S.comp G)
    β : Quiver.Hom (G.obj T) V
    inst✝⁴ : F.IsEquivalence
    inst✝³ : G.Faithful
    inst✝² : G.Full
    inst✝¹ : CategoryTheory.IsIso α
    inst✝ : CategoryTheory.IsIso β
    ⊢ (CategoryTheory.CostructuredArrow.map₂ α β).IsEquivalence
  -/
  apply Comma.isEquivalenceMap
  /-
    🎉 no goals
  -/


/-- `CostructuredArrow.post` is a special case of `CostructuredArrow.map₂` up to natural
isomorphism. -/
def postIsoMap₂ (S : C) (F : B ⥤ C) (G : C ⥤ D) :
    post F G S ≅ map₂ (F := 𝟭 _) (𝟙 (F ⋙ G)) (𝟙 _) :=
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    T T' T'' : D
    Y Y' Y'' : C
    S✝ S' : CategoryTheory.Functor C D
    A : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} A
    B : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} B
    S : C
    F : CategoryTheory.Functor B C
    G : CategoryTheory.Functor C D
    ⊢ ∀ {X Y : CategoryTheory.CostructuredArrow F S} (f : Quiver.Hom X Y), Eq (Cat …
  -/
  NatIso.ofComponents fun _ => isoMk <| Iso.refl _
  /-
    🎉 no goals
  -/


/-- A costructured arrow is called universal if it is terminal. -/
abbrev IsUniversal (f : CostructuredArrow S T) := IsTerminal f


theorem uniq (h : IsUniversal f) (η : g ⟶ f) : η = h.from g :=
  h.hom_ext η (h.from g)


/-- The family of morphisms into a universal arrow. -/
def lift (h : IsUniversal f) (g : CostructuredArrow S T) : g.left ⟶ f.left :=
  (h.from g).left


/-- Any costructured arrow factors through a universal arrow. -/
@[reassoc (attr := simp)]
theorem fac (h : IsUniversal f) (g : CostructuredArrow S T) :
    S.map (h.lift g) ≫ f.hom = g.hom :=
  Category.comp_id g.hom ▸ (h.from g).w


theorem hom_desc (h : IsUniversal f) {c : C} (η : c ⟶ f.left) :
    η = h.lift (mk <| S.map η ≫ f.hom) :=
  let g := mk <| S.map η ≫ f.hom
  congrArg CommaMorphism.left (h.hom_ext (homMk η rfl : g ⟶ f) (h.from g))


/-- Two morphisms into a universal `S`-costructured arrow are equal if their image under `S` are
equal after postcomposing the universal arrow. -/
theorem hom_ext (h : IsUniversal f) {c : C} {η η' : c ⟶ f.left}
    (w : S.map η ≫ f.hom = S.map η' ≫ f.hom) : η = η' := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    T : D
    S : CategoryTheory.Functor C D
    f : CategoryTheory.CostructuredArrow S T
    h : f.IsUniversal
    c : C
    η η' : Quiver.Hom c f.left
    w : Eq (CategoryTheory.CategoryStruct.comp (S.map η) f.hom) (CategoryTheory.Ca …
    ⊢ Eq η η'
  -/
  rw [h.hom_desc η, h.hom_desc η', w]
  /-
    🎉 no goals
  -/


theorem existsUnique (h : IsUniversal f) (g : CostructuredArrow S T) :
    ∃! η : g.left ⟶ f.left, S.map η ≫ f.hom = g.hom :=
                                                /-
                                                  C : Type u₁
                                                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                  D : Type u₂
                                                  inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                  T : D
                                                  S : CategoryTheory.Functor C D
                                                  f✝ : CategoryTheory.CostructuredArrow S T
                                                  h : f✝.IsUniversal
                                                  g : CategoryTheory.CostructuredArrow S T
                                                  f : Quiver.Hom g.left f✝.left
                                                  w : (fun η => Eq (CategoryTheory.CategoryStruct.comp (S.map η) f✝.hom) g.hom) f
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map f) f✝.hom) (CategoryTheory.Cat …
                                                -/
  ⟨h.lift g, h.fac g, fun f w ↦ h.hom_ext <| by simp [w]⟩
                                                /-
                                                  🎉 no goals
                                                -/


/-- Given `X : D` and `F : C ⥤ D`, to upgrade a functor `G : E ⥤ C` to a functor
    `E ⥤ StructuredArrow X F`, it suffices to provide maps `X ⟶ F.obj (G.obj Y)` for all `Y` making
    the obvious triangles involving all `F.map (G.map g)` commute.

    This is of course the same as providing a cone over `F ⋙ G` with cone point `X`, see
    `Functor.toStructuredArrowIsoToStructuredArrow`. -/
@[simps]
def toStructuredArrow (G : E ⥤ C) (X : D) (F : C ⥤ D) (f : (Y : E) → X ⟶ F.obj (G.obj Y))
    (h : ∀ {Y Z : E} (g : Y ⟶ Z), f Y ≫ F.map (G.map g) = f Z) : E ⥤ StructuredArrow X F where
  obj Y := StructuredArrow.mk (f Y)
  map g := StructuredArrow.homMk (G.map g) (h g)


/-- Upgrading a functor `E ⥤ C` to a functor `E ⥤ StructuredArrow X F` and composing with the
    forgetful functor `StructuredArrow X F ⥤ C` recovers the original functor. -/
def toStructuredArrowCompProj (G : E ⥤ C) (X : D) (F : C ⥤ D) (f : (Y : E) → X ⟶ F.obj (G.obj Y))
    (h : ∀ {Y Z : E} (g : Y ⟶ Z), f Y ≫ F.map (G.map g) = f Z) :
    G.toStructuredArrow X F f h ⋙ StructuredArrow.proj _ _ ≅ G :=
  Iso.refl _


@[simp]
lemma toStructuredArrow_comp_proj (G : E ⥤ C) (X : D) (F : C ⥤ D)
    (f : (Y : E) → X ⟶ F.obj (G.obj Y)) (h : ∀ {Y Z : E} (g : Y ⟶ Z), f Y ≫ F.map (G.map g) = f Z) :
    G.toStructuredArrow X F f h ⋙ StructuredArrow.proj _ _ = G :=
  rfl


/-- Given `F : C ⥤ D` and `X : D`, to upgrade a functor `G : E ⥤ C` to a functor
    `E ⥤ CostructuredArrow F X`, it suffices to provide maps `F.obj (G.obj Y) ⟶ X` for all `Y`
    making the obvious triangles involving all `F.map (G.map g)` commute.

    This is of course the same as providing a cocone over `F ⋙ G` with cocone point `X`, see
    `Functor.toCostructuredArrowIsoToCostructuredArrow`. -/
@[simps]
def toCostructuredArrow (G : E ⥤ C) (F : C ⥤ D) (X : D) (f : (Y : E) → F.obj (G.obj Y) ⟶ X)
    (h : ∀ {Y Z : E} (g : Y ⟶ Z), F.map (G.map g) ≫ f Z = f Y) : E ⥤ CostructuredArrow F X where
  obj Y := CostructuredArrow.mk (f Y)
  map g := CostructuredArrow.homMk (G.map g) (h g)


/-- Upgrading a functor `E ⥤ C` to a functor `E ⥤ CostructuredArrow F X` and composing with the
    forgetful functor `CostructuredArrow F X ⥤ C` recovers the original functor. -/
def toCostructuredArrowCompProj (G : E ⥤ C) (F : C ⥤ D) (X : D)
    (f : (Y : E) → F.obj (G.obj Y) ⟶ X) (h : ∀ {Y Z : E} (g : Y ⟶ Z), F.map (G.map g) ≫ f Z = f Y) :
    G.toCostructuredArrow F X f h ⋙ CostructuredArrow.proj _ _ ≅ G :=
  Iso.refl _


@[simp]
lemma toCostructuredArrow_comp_proj (G : E ⥤ C) (F : C ⥤ D) (X : D)
    (f : (Y : E) → F.obj (G.obj Y) ⟶ X) (h : ∀ {Y Z : E} (g : Y ⟶ Z), F.map (G.map g) ≫ f Z = f Y) :
    G.toCostructuredArrow F X f h ⋙ CostructuredArrow.proj _ _ = G :=
rfl


/-- For a functor `F : C ⥤ D` and an object `d : D`, we obtain a contravariant functor from the
category of structured arrows `d ⟶ F.obj c` to the category of costructured arrows
`F.op.obj c ⟶ (op d)`.
-/
@[simps]
def toCostructuredArrow (F : C ⥤ D) (d : D) :
    (StructuredArrow d F)ᵒᵖ ⥤ CostructuredArrow F.op (op d) where
  obj X := @CostructuredArrow.mk _ _ _ _ _ (op X.unop.right) F.op X.unop.hom.op
  map f :=
    CostructuredArrow.homMk f.unop.right.op
      (by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          d : D
          X✝ Y✝ : Opposite (CategoryTheory.StructuredArrow d F)
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.op.map f.unop.right.op) ((fun X => …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          d : D
          X✝ Y✝ : Opposite (CategoryTheory.StructuredArrow d F)
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f.unop.right).op (Opposite.uno …
        -/
        rw [← op_comp, ← f.unop.w, Functor.const_obj_map]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          d : D
          X✝ Y✝ : Opposite (CategoryTheory.StructuredArrow d F)
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id d)  …
        -/
        erw [Category.id_comp])
        /-
          🎉 no goals
        -/


/-- For a functor `F : C ⥤ D` and an object `d : D`, we obtain a contravariant functor from the
category of structured arrows `op d ⟶ F.op.obj c` to the category of costructured arrows
`F.obj c ⟶ d`.
-/
@[simps]
def toCostructuredArrow' (F : C ⥤ D) (d : D) :
    (StructuredArrow (op d) F.op)ᵒᵖ ⥤ CostructuredArrow F d where
  obj X := @CostructuredArrow.mk _ _ _ _ _ (unop X.unop.right) F X.unop.hom.unop
  map f :=
    CostructuredArrow.homMk f.unop.right.unop
      (by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          d : D
          X✝ Y✝ : Opposite (CategoryTheory.StructuredArrow { unop := d } F.op)
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f.unop.right.unop) ((fun X =>  …
        -/
        dsimp
        rw [← Quiver.Hom.unop_op (F.map (Quiver.Hom.unop f.unop.right)), ← unop_comp, ← F.op_map, ←
          f.unop.w, Functor.const_obj_map]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          d : D
          X✝ Y✝ : Opposite (CategoryTheory.StructuredArrow { unop := d } F.op)
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id { u …
        -/
        erw [Category.id_comp])
        /-
          🎉 no goals
        -/


/-- For a functor `F : C ⥤ D` and an object `d : D`, we obtain a contravariant functor from the
category of costructured arrows `F.obj c ⟶ d` to the category of structured arrows
`op d ⟶ F.op.obj c`.
-/
@[simps]
def toStructuredArrow (F : C ⥤ D) (d : D) :
    (CostructuredArrow F d)ᵒᵖ ⥤ StructuredArrow (op d) F.op where
  obj X := @StructuredArrow.mk _ _ _ _ _ (op X.unop.left) F.op X.unop.hom.op
  map f :=
    StructuredArrow.homMk f.unop.left.op
      (by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          d : D
          X✝ Y✝ : Opposite (CategoryTheory.CostructuredArrow F d)
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X => CategoryTheory.StructuredA …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          d : D
          X✝ Y✝ : Opposite (CategoryTheory.CostructuredArrow F d)
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (Opposite.unop X✝).hom.op (F.map f.un …
        -/
        rw [← op_comp, f.unop.w, Functor.const_obj_map]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          d : D
          X✝ Y✝ : Opposite (CategoryTheory.CostructuredArrow F d)
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (Opposite.unop Y✝).hom (CategoryTheor …
        -/
        erw [Category.comp_id])
        /-
          🎉 no goals
        -/


/-- For a functor `F : C ⥤ D` and an object `d : D`, we obtain a contravariant functor from the
category of costructured arrows `F.op.obj c ⟶ op d` to the category of structured arrows
`d ⟶ F.obj c`.
-/
@[simps]
def toStructuredArrow' (F : C ⥤ D) (d : D) :
    (CostructuredArrow F.op (op d))ᵒᵖ ⥤ StructuredArrow d F where
  obj X := @StructuredArrow.mk _ _ _ _ _ (unop X.unop.left) F X.unop.hom.unop
  map f :=
    StructuredArrow.homMk f.unop.left.unop
      (by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          d : D
          X✝ Y✝ : Opposite (CategoryTheory.CostructuredArrow F.op { unop := d })
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X => CategoryTheory.StructuredA …
        -/
        dsimp
        rw [← Quiver.Hom.unop_op (F.map f.unop.left.unop), ← unop_comp, ← F.op_map, f.unop.w,
          Functor.const_obj_map]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          d : D
          X✝ Y✝ : Opposite (CategoryTheory.CostructuredArrow F.op { unop := d })
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (Opposite.unop Y✝).hom (CategoryTheor …
        -/
        erw [Category.comp_id])
        /-
          🎉 no goals
        -/


/-- For a functor `F : C ⥤ D` and an object `d : D`, the category of structured arrows `d ⟶ F.obj c`
is contravariantly equivalent to the category of costructured arrows `F.op.obj c ⟶ op d`.
-/
def structuredArrowOpEquivalence (F : C ⥤ D) (d : D) :
    (StructuredArrow d F)ᵒᵖ ≌ CostructuredArrow F.op (op d) where
  functor := StructuredArrow.toCostructuredArrow F d
  inverse := (CostructuredArrow.toStructuredArrow' F d).rightOp
  unitIso := NatIso.ofComponents
                 /-
                   C : Type u₁
                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝ : CategoryTheory.Category.{v₂, u₂} D
                   F : CategoryTheory.Functor C D
                   d : D
                   X : Opposite (CategoryTheory.StructuredArrow d F)
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (Opposite.unop X).hom (F.map (Categor …
                 -/
      (fun X => (StructuredArrow.isoMk (Iso.refl _)).op)
                 /-
                   🎉 no goals
                 -/
      fun {X Y} f => Quiver.Hom.unop_inj <| by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          d : D
          X Y : Opposite (CategoryTheory.StructuredArrow d F)
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Opposite …
        -/
        apply CommaMorphism.ext <;>
          /-
            case left
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            F : CategoryTheory.Functor C D
            d : D
            X Y : Opposite (CategoryTheory.StructuredArrow d F)
            f : Quiver.Hom X Y
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Opposite …
          -/
          /-
            🎉 no goals
          -/
          dsimp [StructuredArrow.isoMk, Comma.isoMk,StructuredArrow.homMk]; simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
  counitIso := NatIso.ofComponents
                /-
                  C : Type u₁
                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                  D : Type u₂
                  inst✝ : CategoryTheory.Category.{v₂, u₂} D
                  F : CategoryTheory.Functor C D
                  d : D
                  X : CategoryTheory.CostructuredArrow F.op { unop := d }
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.op.map (CategoryTheory.Iso.refl (( …
                -/
      (fun X => CostructuredArrow.isoMk (Iso.refl _))
                /-
                  🎉 no goals
                -/
      fun {X Y} f => by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          d : D
          X Y : CategoryTheory.CostructuredArrow F.op { unop := d }
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.CostructuredArrow.t …
        -/
        apply CommaMorphism.ext <;>
          /-
            case left
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            F : CategoryTheory.Functor C D
            d : D
            X Y : CategoryTheory.CostructuredArrow F.op { unop := d }
            f : Quiver.Hom X Y
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.CostructuredArrow.t …
          -/
          /-
            🎉 no goals
          -/
          dsimp [CostructuredArrow.isoMk, Comma.isoMk, CostructuredArrow.homMk]; simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


/-- For a functor `F : C ⥤ D` and an object `d : D`, the category of costructured arrows
`F.obj c ⟶ d` is contravariantly equivalent to the category of structured arrows
`op d ⟶ F.op.obj c`.
-/
def costructuredArrowOpEquivalence (F : C ⥤ D) (d : D) :
    (CostructuredArrow F d)ᵒᵖ ≌ StructuredArrow (op d) F.op where
  functor := CostructuredArrow.toStructuredArrow F d
  inverse := (StructuredArrow.toCostructuredArrow' F d).rightOp
  unitIso := NatIso.ofComponents
                 /-
                   C : Type u₁
                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝ : CategoryTheory.Category.{v₂, u₂} D
                   F : CategoryTheory.Functor C D
                   d : D
                   X : Opposite (CategoryTheory.CostructuredArrow F d)
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Iso.refl (Oppo …
                 -/
      (fun X => (CostructuredArrow.isoMk (Iso.refl _)).op)
                 /-
                   🎉 no goals
                 -/
      fun {X Y} f => Quiver.Hom.unop_inj <| by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          d : D
          X Y : Opposite (CategoryTheory.CostructuredArrow F d)
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Opposite …
        -/
        apply CommaMorphism.ext <;>
          /-
            case left
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            F : CategoryTheory.Functor C D
            d : D
            X Y : Opposite (CategoryTheory.CostructuredArrow F d)
            f : Quiver.Hom X Y
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Opposite …
          -/
          /-
            🎉 no goals
          -/
          dsimp [CostructuredArrow.isoMk, CostructuredArrow.homMk, Comma.isoMk]; simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
  counitIso := NatIso.ofComponents
                /-
                  C : Type u₁
                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                  D : Type u₂
                  inst✝ : CategoryTheory.Category.{v₂, u₂} D
                  F : CategoryTheory.Functor C D
                  d : D
                  X : CategoryTheory.StructuredArrow { unop := d } F.op
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.StructuredArrow.toC …
                -/
      (fun X => StructuredArrow.isoMk (Iso.refl _))
                /-
                  🎉 no goals
                -/
      fun {X Y} f => by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          d : D
          X Y : CategoryTheory.StructuredArrow { unop := d } F.op
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.StructuredArrow.toC …
        -/
        apply CommaMorphism.ext <;>
          /-
            case left
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            F : CategoryTheory.Functor C D
            d : D
            X Y : CategoryTheory.StructuredArrow { unop := d } F.op
            f : Quiver.Hom X Y
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.StructuredArrow.toC …
          -/
          /-
            🎉 no goals
          -/
          dsimp [StructuredArrow.isoMk, StructuredArrow.homMk, Comma.isoMk]; simp
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- The functor establishing the equivalence `StructuredArrow.preEquivalence`. -/
@[simps!]
def StructuredArrow.preEquivalenceFunctor (f : StructuredArrow e G) :
    StructuredArrow f (pre e F G) ⥤ StructuredArrow f.right F where
  obj g := mk g.hom.right
  map φ := homMk φ.right.right <| by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      e : E
      f : CategoryTheory.StructuredArrow e G
      X✝ Y✝ : CategoryTheory.StructuredArrow f (CategoryTheory.StructuredArrow.pre e …
      φ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun g => CategoryTheory.StructuredA …
    -/
    have := w φ
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      e : E
      f : CategoryTheory.StructuredArrow e G
      X✝ Y✝ : CategoryTheory.StructuredArrow f (CategoryTheory.StructuredArrow.pre e …
      φ : Quiver.Hom X✝ Y✝
      this : Eq (CategoryTheory.CategoryStruct.comp X✝.hom ((CategoryTheory.Structur …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun g => CategoryTheory.StructuredA …
    -/
    simp only [Functor.const_obj_obj] at this ⊢
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      e : E
      f : CategoryTheory.StructuredArrow e G
      X✝ Y✝ : CategoryTheory.StructuredArrow f (CategoryTheory.StructuredArrow.pre e …
      φ : Quiver.Hom X✝ Y✝
      this : Eq (CategoryTheory.CategoryStruct.comp X✝.hom ((CategoryTheory.Structur …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.StructuredArrow.mk X✝ …
    -/
    rw [← this, comp_right]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      e : E
      f : CategoryTheory.StructuredArrow e G
      X✝ Y✝ : CategoryTheory.StructuredArrow f (CategoryTheory.StructuredArrow.pre e …
      φ : Quiver.Hom X✝ Y✝
      this : Eq (CategoryTheory.CategoryStruct.comp X✝.hom ((CategoryTheory.Structur …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.StructuredArrow.mk X✝ …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The inverse functor establishing the equivalence `StructuredArrow.preEquivalence`. -/
@[simps!]
def StructuredArrow.preEquivalenceInverse (f : StructuredArrow e G) :
    StructuredArrow f.right F ⥤ StructuredArrow f (pre e F G) where
  obj g := mk
            (Y := mk (Y := g.right)
              (f.hom ≫ (G.map g.hom : G.obj f.right ⟶ (F ⋙ G).obj g.right)))
             /-
               C : Type u₁
               inst✝² : CategoryTheory.Category.{v₁, u₁} C
               D : Type u₂
               inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
               E : Type u₃
               inst✝ : CategoryTheory.Category.{v₃, u₃} E
               F : CategoryTheory.Functor C D
               G : CategoryTheory.Functor D E
               e : E
               f : CategoryTheory.StructuredArrow e G
               g : CategoryTheory.StructuredArrow f.right F
               ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom (G.map g.hom)) ((CategoryTheory …
             -/
            (homMk g.hom)
             /-
               🎉 no goals
             -/
  map φ := homMk <| homMk φ.right <| by
    simp only [Functor.const_obj_obj, Functor.comp_obj, mk_right, mk_left, mk_hom_eq_self,
      Functor.comp_map, Category.assoc, ← w φ, Functor.map_comp]


/-- A structured arrow category on a `StructuredArrow.pre e F G` functor is equivalent to the
structured arrow category on F -/
@[simps]
def StructuredArrow.preEquivalence (f : StructuredArrow e G) :
    StructuredArrow f (pre e F G) ≌ StructuredArrow f.right F where
  functor := preEquivalenceFunctor F f
  inverse := preEquivalenceInverse F f
                                                  /-
                                                    C : Type u₁
                                                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                    D : Type u₂
                                                    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                    E : Type u₃
                                                    inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                                    F : CategoryTheory.Functor C D
                                                    G : CategoryTheory.Functor D E
                                                    e : E
                                                    f : CategoryTheory.StructuredArrow e G
                                                    x✝ : CategoryTheory.StructuredArrow f (CategoryTheory.StructuredArrow.pre e F G)
                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Category …
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                           /-
                                             🎉 no goals
                                           -/
  unitIso := NatIso.ofComponents (fun _ => isoMk (isoMk (Iso.refl _)))
             /-
               🎉 no goals
             -/
                                             /-
                                               C : Type u₁
                                               inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                               D : Type u₂
                                               inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                               E : Type u₃
                                               inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                               F : CategoryTheory.Functor C D
                                               G : CategoryTheory.Functor D E
                                               e : E
                                               f : CategoryTheory.StructuredArrow e G
                                               x✝ : CategoryTheory.StructuredArrow f.right F
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.StructuredArrow.pre …
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
  counitIso := NatIso.ofComponents (fun _ => isoMk (Iso.refl _))
               /-
                 🎉 no goals
               -/


/-- The functor `StructuredArrow d T ⥤ StructuredArrow e (T ⋙ S)` that `u : e ⟶ S.obj d`
induces via `StructuredArrow.map₂` can be expressed up to isomorphism by
`StructuredArrow.preEquivalence` and `StructuredArrow.proj`. -/
def StructuredArrow.map₂IsoPreEquivalenceInverseCompProj (T : C ⥤ D) (S : D ⥤ E) (d : D) (e : E)
    (u : e ⟶ S.obj d) :
    map₂ (F := 𝟭 _) u (𝟙 (T ⋙ S)) ≅
      (preEquivalence T (mk u)).inverse ⋙ proj (mk u) (pre _ T S) :=
                               /-
                                 C : Type u₁
                                 inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                 D : Type u₂
                                 inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                 E : Type u₃
                                 inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                 F : CategoryTheory.Functor C D
                                 G : CategoryTheory.Functor D E
                                 e✝ : E
                                 T : CategoryTheory.Functor C D
                                 S : CategoryTheory.Functor D E
                                 d : D
                                 e : E
                                 u : Quiver.Hom e (S.obj d)
                                 x✝ : CategoryTheory.StructuredArrow d T
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.StructuredArrow.map₂ …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun _ => isoMk (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- The functor establishing the equivalence `CostructuredArrow.preEquivalence`. -/
@[simps!]
def CostructuredArrow.preEquivalence.functor (f : CostructuredArrow G e) :
    CostructuredArrow (pre F G e) f ⥤ CostructuredArrow F f.left where
  obj g := mk g.hom.left
  map φ := homMk φ.left.left <| by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      e : E
      f : CategoryTheory.CostructuredArrow G e
      X✝ Y✝ : CategoryTheory.CostructuredArrow (CategoryTheory.CostructuredArrow.pre …
      φ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ.left.left) ((fun g => Catego …
    -/
    have := w φ
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      e : E
      f : CategoryTheory.CostructuredArrow G e
      X✝ Y✝ : CategoryTheory.CostructuredArrow (CategoryTheory.CostructuredArrow.pre …
      φ : Quiver.Hom X✝ Y✝
      this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CostructuredArr …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ.left.left) ((fun g => Catego …
    -/
    simp only [Functor.const_obj_obj] at this ⊢
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      e : E
      f : CategoryTheory.CostructuredArrow G e
      X✝ Y✝ : CategoryTheory.CostructuredArrow (CategoryTheory.CostructuredArrow.pre …
      φ : Quiver.Hom X✝ Y✝
      this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CostructuredArr …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ.left.left) (CategoryTheory.C …
    -/
    rw [← this, comp_left]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D E
      e : E
      f : CategoryTheory.CostructuredArrow G e
      X✝ Y✝ : CategoryTheory.CostructuredArrow (CategoryTheory.CostructuredArrow.pre …
      φ : Quiver.Hom X✝ Y✝
      this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CostructuredArr …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ.left.left) (CategoryTheory.C …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The inverse functor establishing the equivalence `CostructuredArrow.preEquivalence`. -/
@[simps!]
def CostructuredArrow.preEquivalence.inverse (f : CostructuredArrow G e) :
    CostructuredArrow F f.left ⥤ CostructuredArrow (pre F G e) f where
                                                             /-
                                                               C : Type u₁
                                                               inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                               D : Type u₂
                                                               inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                               E : Type u₃
                                                               inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                                               F : CategoryTheory.Functor C D
                                                               G : CategoryTheory.Functor D E
                                                               e : E
                                                               f : CategoryTheory.CostructuredArrow G e
                                                               g : CategoryTheory.CostructuredArrow F f.left
                                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map g.hom) f.hom) ((CategoryTheory …
                                                             -/
  obj g := mk (Y := mk (Y := g.left) (G.map g.hom ≫ f.hom)) (homMk g.hom)
                                                             /-
                                                               🎉 no goals
                                                             -/
  map φ := homMk <| homMk φ.left <| by
    simp only [Functor.const_obj_obj, Functor.comp_obj, mk_left, Functor.comp_map, mk_hom_eq_self,
      ← w φ, Functor.map_comp, Category.assoc]


/-- A costructured arrow category on a `CostructuredArrow.pre F G e` functor is equivalent to the
costructured arrow category on F -/
def CostructuredArrow.preEquivalence (f : CostructuredArrow G e) :
    CostructuredArrow (pre F G e) f ≌ CostructuredArrow F f.left where
  functor := preEquivalence.functor F f
  inverse := preEquivalence.inverse F f
                                                  /-
                                                    C : Type u₁
                                                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                    D : Type u₂
                                                    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                    E : Type u₃
                                                    inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                                    F : CategoryTheory.Functor C D
                                                    G : CategoryTheory.Functor D E
                                                    e : E
                                                    f : CategoryTheory.CostructuredArrow G e
                                                    x✝ : CategoryTheory.CostructuredArrow (CategoryTheory.CostructuredArrow.pre F  …
                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.comp G).map (CategoryTheory.Iso.r …
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                           /-
                                             🎉 no goals
                                           -/
  unitIso := NatIso.ofComponents (fun _ => isoMk (isoMk (Iso.refl _)))
             /-
               🎉 no goals
             -/
                                             /-
                                               C : Type u₁
                                               inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                               D : Type u₂
                                               inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                               E : Type u₃
                                               inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                               F : CategoryTheory.Functor C D
                                               G : CategoryTheory.Functor D E
                                               e : E
                                               f : CategoryTheory.CostructuredArrow G e
                                               x✝ : CategoryTheory.CostructuredArrow F f.left
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Iso.refl (((Ca …
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
  counitIso := NatIso.ofComponents (fun _ => isoMk (Iso.refl _))
               /-
                 🎉 no goals
               -/


/-- The functor `CostructuredArrow T d ⥤ CostructuredArrow (T ⋙ S) e` that `u : S.obj d ⟶ e`
induces via `CostructuredArrow.map₂` can be expressed up to isomorphism by
`CostructuredArrow.preEquivalence` and `CostructuredArrow.proj`. -/
def CostructuredArrow.map₂IsoPreEquivalenceInverseCompProj (T : C ⥤ D) (S : D ⥤ E) (d : D) (e : E)
    (u : S.obj d ⟶ e) :
    map₂ (F := 𝟭 _) (U := T ⋙ S) (𝟙 (T ⋙ S)) u ≅
      (preEquivalence T (mk u)).inverse ⋙ proj (pre T S _) (mk u) :=
                               /-
                                 C : Type u₁
                                 inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                 D : Type u₂
                                 inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                 E : Type u₃
                                 inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                 F : CategoryTheory.Functor C D
                                 G : CategoryTheory.Functor D E
                                 e✝ : E
                                 T : CategoryTheory.Functor C D
                                 S : CategoryTheory.Functor D E
                                 d : D
                                 e : E
                                 u : Quiver.Hom (S.obj d) e
                                 x✝ : CategoryTheory.CostructuredArrow T d
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ((T.comp S).map (CategoryTheory.Iso.r …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun _ => isoMk (Iso.refl _)
  /-
    🎉 no goals
  -/



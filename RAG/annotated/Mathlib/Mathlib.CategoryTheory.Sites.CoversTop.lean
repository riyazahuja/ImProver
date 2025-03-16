/-- A family of objects `Y : I → C` "covers the final object"
if for all `X : C`, the sieve `ofObjects Y X` is a covering sieve. -/
def CoversTop {I : Type*} (Y : I → C) : Prop :=
  ∀ (X : C), Sieve.ofObjects Y X ∈ J X


lemma coversTop_iff_of_isTerminal (X : C) (hX : IsTerminal X)
    {I : Type*} (Y : I → C) :
    J.CoversTop Y ↔ Sieve.ofObjects Y X ∈ J X := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    X : C
    hX : CategoryTheory.Limits.IsTerminal X
    I : Type u_1
    Y : I → C
    ⊢ Iff (J.CoversTop Y) (Membership.mem (J X) (CategoryTheory.Sieve.ofObjects Y  …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      hX : CategoryTheory.Limits.IsTerminal X
      I : Type u_1
      Y : I → C
      ⊢ J.CoversTop Y → Membership.mem (J X) (CategoryTheory.Sieve.ofObjects Y X)
    -/
  · tauto
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      hX : CategoryTheory.Limits.IsTerminal X
      I : Type u_1
      Y : I → C
      ⊢ Membership.mem (J X) (CategoryTheory.Sieve.ofObjects Y X) → J.CoversTop Y
    -/
  · intro h W
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      hX : CategoryTheory.Limits.IsTerminal X
      I : Type u_1
      Y : I → C
      h : Membership.mem (J X) (CategoryTheory.Sieve.ofObjects Y X)
      W : C
      ⊢ Membership.mem (J W) (CategoryTheory.Sieve.ofObjects Y W)
    -/
    apply J.superset_covering _ (J.pullback_stable (hX.from W) h)
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      hX : CategoryTheory.Limits.IsTerminal X
      I : Type u_1
      Y : I → C
      h : Membership.mem (J X) (CategoryTheory.Sieve.ofObjects Y X)
      W : C
      ⊢ LE.le (CategoryTheory.Sieve.pullback (hX.from W) (CategoryTheory.Sieve.ofObj …
    -/
    rintro T a ⟨i, ⟨b⟩⟩
    /-
      case intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      X : C
      hX : CategoryTheory.Limits.IsTerminal X
      I : Type u_1
      Y : I → C
      h : Membership.mem (J X) (CategoryTheory.Sieve.ofObjects Y X)
      W T : C
      a : Quiver.Hom T W
      i : I
      b : Quiver.Hom T (Y i)
      ⊢ (CategoryTheory.Sieve.ofObjects Y W).arrows a
    -/
    exact ⟨i, ⟨b⟩⟩
    /-
      🎉 no goals
    -/


/-- The cover of any object `W : C` attached to a family of objects `Y` that satisfy
`J.CoversTop Y` -/
abbrev cover (W : C) : Cover J W := ⟨Sieve.ofObjects Y W, hY W⟩


lemma ext (F : Sheaf J A) {c : Cone F.1} (hc : IsLimit c) {X : A} {f g : X ⟶ c.pt}
    (h : ∀ (i : I), f ≫ c.π.app (Opposite.op (Y i)) =
      g ≫ c.π.app (Opposite.op (Y i))) :
    f = g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} A
    I : Type u_1
    Y : I → C
    hY : J.CoversTop Y
    F : CategoryTheory.Sheaf J A
    c : CategoryTheory.Limits.Cone F.val
    hc : CategoryTheory.Limits.IsLimit c
    X : A
    f g : Quiver.Hom X c.pt
    h : ∀ (i : I), Eq (CategoryTheory.CategoryStruct.comp f (c.π.app { unop := Y i …
    ⊢ Eq f g
  -/
  refine hc.hom_ext (fun Z => F.2.hom_ext (hY.cover Z.unop) _ _ ?_)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} A
    I : Type u_1
    Y : I → C
    hY : J.CoversTop Y
    F : CategoryTheory.Sheaf J A
    c : CategoryTheory.Limits.Cone F.val
    hc : CategoryTheory.Limits.IsLimit c
    X : A
    f g : Quiver.Hom X c.pt
    h : ∀ (i : I), Eq (CategoryTheory.CategoryStruct.comp f (c.π.app { unop := Y i …
    Z : Opposite C
    ⊢ ∀ (I_1 : (hY.cover (Opposite.unop Z)).Arrow), Eq (CategoryTheory.CategoryStr …
  -/
  rintro ⟨W, a, ⟨i, ⟨b⟩⟩⟩
  /-
    case mk.intro.intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} A
    I : Type u_1
    Y : I → C
    hY : J.CoversTop Y
    F : CategoryTheory.Sheaf J A
    c : CategoryTheory.Limits.Cone F.val
    hc : CategoryTheory.Limits.IsLimit c
    X : A
    f g : Quiver.Hom X c.pt
    h : ∀ (i : I), Eq (CategoryTheory.CategoryStruct.comp f (c.π.app { unop := Y i …
    Z : Opposite C
    W : C
    a : Quiver.Hom W (Opposite.unop Z)
    i : I
    b : Quiver.Hom W (Y i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
  -/
  simpa using h i =≫ F.1.map b.op
  /-
    🎉 no goals
  -/


lemma sections_ext (F : Sheaf J (Type _)) {x y : F.1.sections}
    (h : ∀ (i : I), x.1 (Opposite.op (Y i)) = y.1 (Opposite.op (Y i))) :
    x = y := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    I : Type u_1
    Y : I → C
    hY : J.CoversTop Y
    F : CategoryTheory.Sheaf J (Type u_2)
    x y : ↑F.val.sections
    h : ∀ (i : I), Eq (↑x { unop := Y i }) (↑y { unop := Y i })
    ⊢ Eq x y
  -/
  ext W
  apply (Presieve.isSeparated_of_isSheaf J F.1
    ((isSheaf_iff_isSheaf_of_type _ _).1 F.2) _ (hY W.unop)).ext
  /-
    case a.h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    I : Type u_1
    Y : I → C
    hY : J.CoversTop Y
    F : CategoryTheory.Sheaf J (Type u_2)
    x y : ↑F.val.sections
    h : ∀ (i : I), Eq (↑x { unop := Y i }) (↑y { unop := Y i })
    W : Opposite C
    ⊢ ∀ ⦃Y_1 : C⦄ ⦃f : Quiver.Hom Y_1 (Opposite.unop W)⦄, (CategoryTheory.Sieve.of …
  -/
  rintro T a ⟨i, ⟨b⟩⟩
  /-
    case a.h.intro.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    I : Type u_1
    Y : I → C
    hY : J.CoversTop Y
    F : CategoryTheory.Sheaf J (Type u_2)
    x y : ↑F.val.sections
    h : ∀ (i : I), Eq (↑x { unop := Y i }) (↑y { unop := Y i })
    W : Opposite C
    T : C
    a : Quiver.Hom T (Opposite.unop W)
    i : I
    b : Quiver.Hom T (Y i)
    ⊢ Eq (F.val.map a.op (↑x W)) (F.val.map a.op (↑y W))
  -/
  simpa using congr_arg (F.1.map b.op) (h i)
  /-
    🎉 no goals
  -/


/-- A family of elements of a presheaf of types `F` indexed by a family of objects
`Y : I → C` consists of the data of an element in `F.obj (Opposite.op (Y i))` for all `i`. -/
def FamilyOfElementsOnObjects := ∀ (i : I), F.obj (Opposite.op (Y i))


/-- `x : FamilyOfElementsOnObjects F Y` is compatible if for any object `Z` such that
there exists a morphism `f : Z → Y i`, then the pullback of `x i` by `f` is independent
of `f` and `i`. -/
def IsCompatible (x : FamilyOfElementsOnObjects F Y) : Prop :=
  ∀ (Z : C) (i j : I) (f : Z ⟶ Y i) (g : Z ⟶ Y j),
    F.map f.op (x i) = F.map g.op (x j)


/-- A family of elements indexed by `Sieve.ofObjects Y X` that is induced by
`x : FamilyOfElementsOnObjects F Y`. See the equational lemma
`IsCompatible.familyOfElements_apply` which holds under the assumption `x.IsCompatible`. -/
noncomputable def familyOfElements (X : C) :
    Presieve.FamilyOfElements F (Sieve.ofObjects Y X).arrows :=
  fun _ _ hf => F.map hf.choose_spec.some.op (x _)


lemma familyOfElements_apply (hx : x.IsCompatible) {X Z : C} (f : Z ⟶ X) (i : I) (φ : Z ⟶ Y i) :
    familyOfElements x X f ⟨i, ⟨φ⟩⟩ = F.map φ.op (x i) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    I : Type u_1
    Y : I → C
    x : CategoryTheory.Presheaf.FamilyOfElementsOnObjects F Y
    hx : x.IsCompatible
    X Z : C
    f : Quiver.Hom Z X
    i : I
    φ : Quiver.Hom Z (Y i)
    ⊢ Eq (x.familyOfElements X f ⋯) (F.map φ.op (x i))
  -/
  apply hx
  /-
    🎉 no goals
  -/


lemma familyOfElements_isCompatible (hx : x.IsCompatible) (X : C) :
    (familyOfElements x X).Compatible := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    I : Type u_1
    Y : I → C
    x : CategoryTheory.Presheaf.FamilyOfElementsOnObjects F Y
    hx : x.IsCompatible
    X : C
    ⊢ (x.familyOfElements X).Compatible
  -/
  intro Y₁ Y₂ Z g₁ g₂ f₁ f₂ ⟨i₁, ⟨φ₁⟩⟩ ⟨i₂, ⟨φ₂⟩⟩ _
  simpa [hx.familyOfElements_apply f₁ i₁ φ₁,
    hx.familyOfElements_apply f₂ i₂ φ₂] using hx Z i₁ i₂ (g₁ ≫ φ₁) (g₂ ≫ φ₂)


lemma existsUnique_section (hx : x.IsCompatible) (hY : J.CoversTop Y) (hF : IsSheaf J F) :
    ∃! (s : F.sections), ∀ (i : I), s.1 (Opposite.op (Y i)) = x i := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    I : Type u_1
    Y : I → C
    x : CategoryTheory.Presheaf.FamilyOfElementsOnObjects F Y
    hx : x.IsCompatible
    hY : J.CoversTop Y
    hF : CategoryTheory.Presheaf.IsSheaf J F
    ⊢ ExistsUnique fun s => ∀ (i : I), Eq (↑s { unop := Y i }) (x i)
  -/
  have H := (isSheaf_iff_isSheaf_of_type _ _).1 hF
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    I : Type u_1
    Y : I → C
    x : CategoryTheory.Presheaf.FamilyOfElementsOnObjects F Y
    hx : x.IsCompatible
    hY : J.CoversTop Y
    hF : CategoryTheory.Presheaf.IsSheaf J F
    H : CategoryTheory.Presieve.IsSheaf J F
    ⊢ ExistsUnique fun s => ∀ (i : I), Eq (↑s { unop := Y i }) (x i)
  -/
  apply existsUnique_of_exists_of_unique
  · let s := fun (X : C) => (H _ (hY X)).amalgamate _
      (hx.familyOfElements_isCompatible X)
    have hs : ∀ {X : C} (i : I) (f : X ⟶ Y i), s X = F.map f.op (x i) := fun {X} i f => by
      have h := Presieve.IsSheafFor.valid_glue (H _ (hY X))
          (hx.familyOfElements_isCompatible _) (𝟙 _) ⟨i, ⟨f⟩⟩
      simp only [op_id, F.map_id, types_id_apply] at h
      exact h.trans (hx.familyOfElements_apply _ _ _)
    have hs' : ∀ {W X : C} (a : W ⟶ X) (i : I) (_ : W ⟶ Y i), F.map a.op (s X) = s W := by
      intro W X a i b
      rw [hs i b]
      exact (Presieve.IsSheafFor.valid_glue (H _ (hY X))
        (hx.familyOfElements_isCompatible _) a ⟨i, ⟨b⟩⟩).trans (familyOfElements_apply hx _ _ _)
    /-
      case hex
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      I : Type u_1
      Y : I → C
      x : CategoryTheory.Presheaf.FamilyOfElementsOnObjects F Y
      hx : x.IsCompatible
      hY : J.CoversTop Y
      hF : CategoryTheory.Presheaf.IsSheaf J F
      H : CategoryTheory.Presieve.IsSheaf J F
      s : (X : C) → F.obj { unop := X } := fun X => ⋯.amalgamate (x.familyOfElements …
      hs : ∀ {X : C} (i : I) (f : Quiver.Hom X (Y i)), Eq (s X) (F.map f.op (x i))
      hs' : ∀ {W X : C} (a : Quiver.Hom W X) (i : I), Quiver.Hom W (Y i) → Eq (F.map …
      ⊢ Exists fun x_1 => ∀ (i : I), Eq (↑x_1 { unop := Y i }) (x i)
    -/
    refine ⟨⟨fun X => s X.unop, ?_⟩, fun i => (hs i (𝟙 (Y i))).trans (by simp)⟩
    /-
      case hex
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      I : Type u_1
      Y : I → C
      x : CategoryTheory.Presheaf.FamilyOfElementsOnObjects F Y
      hx : x.IsCompatible
      hY : J.CoversTop Y
      hF : CategoryTheory.Presheaf.IsSheaf J F
      H : CategoryTheory.Presieve.IsSheaf J F
      s : (X : C) → F.obj { unop := X } := fun X => ⋯.amalgamate (x.familyOfElements …
      hs : ∀ {X : C} (i : I) (f : Quiver.Hom X (Y i)), Eq (s X) (F.map f.op (x i))
      hs' : ∀ {W X : C} (a : Quiver.Hom W X) (i : I), Quiver.Hom W (Y i) → Eq (F.map …
      ⊢ Membership.mem F.sections fun X => s (Opposite.unop X)
    -/
    rintro ⟨Y₁⟩ ⟨Y₂⟩ ⟨f : Y₂ ⟶ Y₁⟩
    /-
      case hex.op.op.op
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      I : Type u_1
      Y : I → C
      x : CategoryTheory.Presheaf.FamilyOfElementsOnObjects F Y
      hx : x.IsCompatible
      hY : J.CoversTop Y
      hF : CategoryTheory.Presheaf.IsSheaf J F
      H : CategoryTheory.Presieve.IsSheaf J F
      s : (X : C) → F.obj { unop := X } := fun X => ⋯.amalgamate (x.familyOfElements …
      hs : ∀ {X : C} (i : I) (f : Quiver.Hom X (Y i)), Eq (s X) (F.map f.op (x i))
      hs' : ∀ {W X : C} (a : Quiver.Hom W X) (i : I), Quiver.Hom W (Y i) → Eq (F.map …
      Y₁ Y₂ : C
      f : Quiver.Hom Y₂ Y₁
      ⊢ Eq (F.map { unop := f } ((fun X => s (Opposite.unop X)) { unop := Y₁ })) ((f …
    -/
    change F.map f.op (s Y₁) = s Y₂
    /-
      case hex.op.op.op
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      I : Type u_1
      Y : I → C
      x : CategoryTheory.Presheaf.FamilyOfElementsOnObjects F Y
      hx : x.IsCompatible
      hY : J.CoversTop Y
      hF : CategoryTheory.Presheaf.IsSheaf J F
      H : CategoryTheory.Presieve.IsSheaf J F
      s : (X : C) → F.obj { unop := X } := fun X => ⋯.amalgamate (x.familyOfElements …
      hs : ∀ {X : C} (i : I) (f : Quiver.Hom X (Y i)), Eq (s X) (F.map f.op (x i))
      hs' : ∀ {W X : C} (a : Quiver.Hom W X) (i : I), Quiver.Hom W (Y i) → Eq (F.map …
      Y₁ Y₂ : C
      f : Quiver.Hom Y₂ Y₁
      ⊢ Eq (F.map f.op (s Y₁)) (s Y₂)
    -/
    apply (Presieve.isSeparated_of_isSheaf J F H _ (hY Y₂)).ext
    /-
      case hex.op.op.op
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      I : Type u_1
      Y : I → C
      x : CategoryTheory.Presheaf.FamilyOfElementsOnObjects F Y
      hx : x.IsCompatible
      hY : J.CoversTop Y
      hF : CategoryTheory.Presheaf.IsSheaf J F
      H : CategoryTheory.Presieve.IsSheaf J F
      s : (X : C) → F.obj { unop := X } := fun X => ⋯.amalgamate (x.familyOfElements …
      hs : ∀ {X : C} (i : I) (f : Quiver.Hom X (Y i)), Eq (s X) (F.map f.op (x i))
      hs' : ∀ {W X : C} (a : Quiver.Hom W X) (i : I), Quiver.Hom W (Y i) → Eq (F.map …
      Y₁ Y₂ : C
      f : Quiver.Hom Y₂ Y₁
      ⊢ ∀ ⦃Y_1 : C⦄ ⦃f_1 : Quiver.Hom Y_1 Y₂⦄, (CategoryTheory.Sieve.ofObjects Y Y₂) …
    -/
    rintro Z φ ⟨i, ⟨g⟩⟩
    /-
      case hex.op.op.op.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      I : Type u_1
      Y : I → C
      x : CategoryTheory.Presheaf.FamilyOfElementsOnObjects F Y
      hx : x.IsCompatible
      hY : J.CoversTop Y
      hF : CategoryTheory.Presheaf.IsSheaf J F
      H : CategoryTheory.Presieve.IsSheaf J F
      s : (X : C) → F.obj { unop := X } := fun X => ⋯.amalgamate (x.familyOfElements …
      hs : ∀ {X : C} (i : I) (f : Quiver.Hom X (Y i)), Eq (s X) (F.map f.op (x i))
      hs' : ∀ {W X : C} (a : Quiver.Hom W X) (i : I), Quiver.Hom W (Y i) → Eq (F.map …
      Y₁ Y₂ : C
      f : Quiver.Hom Y₂ Y₁
      Z : C
      φ : Quiver.Hom Z Y₂
      i : I
      g : Quiver.Hom Z (Y i)
      ⊢ Eq (F.map φ.op (F.map f.op (s Y₁))) (F.map φ.op (s Y₂))
    -/
    rw [hs' φ i g, ← hs' (φ ≫ f) i g, op_comp, F.map_comp]
    /-
      case hex.op.op.op.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      I : Type u_1
      Y : I → C
      x : CategoryTheory.Presheaf.FamilyOfElementsOnObjects F Y
      hx : x.IsCompatible
      hY : J.CoversTop Y
      hF : CategoryTheory.Presheaf.IsSheaf J F
      H : CategoryTheory.Presieve.IsSheaf J F
      s : (X : C) → F.obj { unop := X } := fun X => ⋯.amalgamate (x.familyOfElements …
      hs : ∀ {X : C} (i : I) (f : Quiver.Hom X (Y i)), Eq (s X) (F.map f.op (x i))
      hs' : ∀ {W X : C} (a : Quiver.Hom W X) (i : I), Quiver.Hom W (Y i) → Eq (F.map …
      Y₁ Y₂ : C
      f : Quiver.Hom Y₂ Y₁
      Z : C
      φ : Quiver.Hom Z Y₂
      i : I
      g : Quiver.Hom Z (Y i)
      ⊢ Eq (F.map φ.op (F.map f.op (s Y₁))) (CategoryTheory.CategoryStruct.comp (F.m …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case hunique
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      I : Type u_1
      Y : I → C
      x : CategoryTheory.Presheaf.FamilyOfElementsOnObjects F Y
      hx : x.IsCompatible
      hY : J.CoversTop Y
      hF : CategoryTheory.Presheaf.IsSheaf J F
      H : CategoryTheory.Presieve.IsSheaf J F
      ⊢ ∀ (y₁ y₂ : ↑F.sections), (∀ (i : I), Eq (↑y₁ { unop := Y i }) (x i)) → (∀ (i …
    -/
  · intro y₁ y₂ hy₁ hy₂
    /-
      case hunique
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      I : Type u_1
      Y : I → C
      x : CategoryTheory.Presheaf.FamilyOfElementsOnObjects F Y
      hx : x.IsCompatible
      hY : J.CoversTop Y
      hF : CategoryTheory.Presheaf.IsSheaf J F
      H : CategoryTheory.Presieve.IsSheaf J F
      y₁ y₂ : ↑F.sections
      hy₁ : ∀ (i : I), Eq (↑y₁ { unop := Y i }) (x i)
      hy₂ : ∀ (i : I), Eq (↑y₂ { unop := Y i }) (x i)
      ⊢ Eq y₁ y₂
    -/
    exact hY.sections_ext ⟨F, hF⟩ (fun i => by rw [hy₁, hy₂])
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-12-17")] alias exists_unique_section := existsUnique_section


/-- The section of a sheaf of types which lifts a compatible family of elements indexed
by objects which cover the terminal object. -/
noncomputable def section_ : F.sections := (hx.existsUnique_section hY hF).choose


@[simp]
lemma section_apply (i : I) : (hx.section_ hY hF).1 (Opposite.op (Y i)) = x i :=
  (hx.existsUnique_section hY hF).choose_spec.1 i



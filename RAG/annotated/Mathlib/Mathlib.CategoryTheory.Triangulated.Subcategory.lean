/-- A triangulated subcategory of a pretriangulated category `C` consists of
a predicate `P : C → Prop` which contains a zero object, is stable by shifts, and such that
if `X₁ ⟶ X₂ ⟶ X₃ ⟶ X₁⟦1⟧` is a distinguished triangle such that if `X₁` and `X₃` satisfy
`P` then `X₂` is isomorphic to an object satisfying `P`. -/
structure Subcategory where
  /-- the underlying predicate on objects of a triangulated subcategory -/
  P : C → Prop
  zero' : ∃ (Z : C) (_ : IsZero Z), P Z
  shift (X : C) (n : ℤ) : P X → P (X⟦n⟧)
  ext₂' (T : Triangle C) (_ : T ∈ distTriang C) : P T.obj₁ → P T.obj₃ → isoClosure P T.obj₂


lemma zero [ClosedUnderIsomorphisms S.P] : S.P 0 := by
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝¹ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    inst✝ : CategoryTheory.ClosedUnderIsomorphisms S.P
    ⊢ S.P 0
  -/
  obtain ⟨X, hX, mem⟩ := S.zero'
  /-
    case intro.intro
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝¹ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    inst✝ : CategoryTheory.ClosedUnderIsomorphisms S.P
    X : C
    hX : CategoryTheory.Limits.IsZero X
    mem : S.P X
    ⊢ S.P 0
  -/
  exact mem_of_iso _ hX.isoZero mem
  /-
    🎉 no goals
  -/


/-- The closure under isomorphisms of a triangulated subcategory. -/
def isoClosure : Subcategory C where
  P := CategoryTheory.isoClosure S.P
  zero' := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.4072, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      ⊢ Exists fun Z => Exists fun x => CategoryTheory.isoClosure S.P Z
    -/
    obtain ⟨Z, hZ, hZ'⟩ := S.zero'
    /-
      case intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.4072, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      Z : C
      hZ : CategoryTheory.Limits.IsZero Z
      hZ' : S.P Z
      ⊢ Exists fun Z => Exists fun x => CategoryTheory.isoClosure S.P Z
    -/
    exact ⟨Z, hZ, Z, hZ', ⟨Iso.refl _⟩⟩
    /-
      🎉 no goals
    -/
  shift X n := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.4072, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      X : C
      n : Int
      ⊢ CategoryTheory.isoClosure S.P X → CategoryTheory.isoClosure S.P ((CategoryTh …
    -/
    rintro ⟨Y, hY, ⟨e⟩⟩
    /-
      case intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.4072, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      X : C
      n : Int
      Y : C
      hY : S.P Y
      e : CategoryTheory.Iso X Y
      ⊢ CategoryTheory.isoClosure S.P ((CategoryTheory.shiftFunctor C n).obj X)
    -/
    exact ⟨Y⟦n⟧, S.shift Y n hY, ⟨(shiftFunctor C n).mapIso e⟩⟩
    /-
      🎉 no goals
    -/
  ext₂' := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.4072, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      ⊢ ∀ (T : CategoryTheory.Pretriangulated.Triangle C), Membership.mem CategoryTh …
    -/
    rintro T hT ⟨X₁, h₁, ⟨e₁⟩⟩ ⟨X₃, h₃, ⟨e₃⟩⟩
    exact le_isoClosure _ _
      (S.ext₂' (Triangle.mk (e₁.inv ≫ T.mor₁) (T.mor₂ ≫ e₃.hom) (e₃.inv ≫ T.mor₃ ≫ e₁.hom⟦1⟧'))
      (isomorphic_distinguished _ hT _
        (Triangle.isoMk _ _ e₁.symm (Iso.refl _) e₃.symm (by aesop_cat) (by aesop_cat) (by
          dsimp
          simp only [assoc, Iso.cancel_iso_inv_left, ← Functor.map_comp, e₁.hom_inv_id,
            Functor.map_id, comp_id]))) h₁ h₃)


instance : ClosedUnderIsomorphisms S.isoClosure.P := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : CategoryTheory.HasShift C Int
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    ⊢ CategoryTheory.ClosedUnderIsomorphisms S.isoClosure.P
  -/
  dsimp only [isoClosure]
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : CategoryTheory.HasShift C Int
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    ⊢ CategoryTheory.ClosedUnderIsomorphisms (CategoryTheory.isoClosure S.P)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- An alternative constructor for "strictly full" triangulated subcategory. -/
def mk' : Subcategory C where
  P := P
  zero' := ⟨0, isZero_zero _, zero⟩
  shift := shift
  ext₂' T hT h₁ h₃ := le_isoClosure P _ (ext₂ T hT h₁ h₃)


instance : ClosedUnderIsomorphisms (mk' P zero shift ext₂).P where
  of_iso {X Y} e hX := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      P : C → Prop
      zero : P 0
      shift : ∀ (X : C) (n : Int), P X → P ((CategoryTheory.shiftFunctor C n).obj X)
      ext₂ : ∀ (T : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Categ …
      X Y : C
      e : CategoryTheory.Iso X Y
      hX : (CategoryTheory.Triangulated.Subcategory.mk' P zero shift ext₂).P X
      ⊢ (CategoryTheory.Triangulated.Subcategory.mk' P zero shift ext₂).P Y
    -/
    refine ext₂ (Triangle.mk e.hom (0 : Y ⟶ 0) 0) ?_ hX zero
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      P : C → Prop
      zero : P 0
      shift : ∀ (X : C) (n : Int), P X → P ((CategoryTheory.shiftFunctor C n).obj X)
      ext₂ : ∀ (T : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Categ …
      X Y : C
      e : CategoryTheory.Iso X Y
      hX : (CategoryTheory.Triangulated.Subcategory.mk' P zero shift ext₂).P X
      ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Catego …
    -/
    refine isomorphic_distinguished _ (contractible_distinguished X) _ ?_
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      P : C → Prop
      zero : P 0
      shift : ∀ (X : C) (n : Int), P X → P ((CategoryTheory.shiftFunctor C n).obj X)
      ext₂ : ∀ (T : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Categ …
      X Y : C
      e : CategoryTheory.Iso X Y
      hX : (CategoryTheory.Triangulated.Subcategory.mk' P zero shift ext₂).P X
      ⊢ CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk e.hom 0 0) (C …
    -/
    exact Triangle.isoMk _ _ (Iso.refl _) e.symm (Iso.refl _)
    /-
      🎉 no goals
    -/


lemma ext₂ [ClosedUnderIsomorphisms S.P]
    (T : Triangle C) (hT : T ∈ distTriang C) (h₁ : S.P T.obj₁)
    (h₃ : S.P T.obj₃) : S.P T.obj₂ := by
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝¹ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    inst✝ : CategoryTheory.ClosedUnderIsomorphisms S.P
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    h₁ : S.P T.obj₁
    h₃ : S.P T.obj₃
    ⊢ S.P T.obj₂
  -/
  simpa only [isoClosure_eq_self] using S.ext₂' T hT h₁ h₃
  /-
    🎉 no goals
  -/


/-- Given `S : Triangulated.Subcategory C`, this is the class of morphisms on `C` which
consists of morphisms whose cone satisfies `S.P`. -/
def W : MorphismProperty C := fun X Y f => ∃ (Z : C) (g : Y ⟶ Z) (h : Z ⟶ X⟦(1 : ℤ)⟧)
  (_ : Triangle.mk f g h ∈ distTriang C), S.P Z


lemma W_iff {X Y : C} (f : X ⟶ Y) :
    S.W f ↔ ∃ (Z : C) (g : Y ⟶ Z) (h : Z ⟶ X⟦(1 : ℤ)⟧)
                                                          /-
                                                            C : Type u_1
                                                            inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
                                                            inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
                                                            inst✝³ : CategoryTheory.HasShift C Int
                                                            inst✝² : CategoryTheory.Preadditive C
                                                            inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
                                                            inst✝ : CategoryTheory.Pretriangulated C
                                                            S : CategoryTheory.Triangulated.Subcategory C
                                                            X Y : C
                                                            f : Quiver.Hom X Y
                                                            ⊢ Iff (S.W f) (Exists fun Z => Exists fun g => Exists fun h => Exists fun x => …
                                                          -/
      (_ : Triangle.mk f g h ∈ distTriang C), S.P Z := by rfl
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma W_iff' {Y Z : C} (g : Y ⟶ Z) :
    S.W g ↔ ∃ (X : C) (f : X ⟶ Y) (h : Z ⟶ X⟦(1 : ℤ)⟧)
      (_ : Triangle.mk f g h ∈ distTriang C), S.P X := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : CategoryTheory.HasShift C Int
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    Y Z : C
    g : Quiver.Hom Y Z
    ⊢ Iff (S.W g) (Exists fun X => Exists fun f => Exists fun h => Exists fun x => …
  -/
  rw [S.W_iff]
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : CategoryTheory.HasShift C Int
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    Y Z : C
    g : Quiver.Hom Y Z
    ⊢ Iff (Exists fun Z_1 => Exists fun g_1 => Exists fun h => Exists fun x => S.P …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      Y Z : C
      g : Quiver.Hom Y Z
      ⊢ (Exists fun Z_1 => Exists fun g_1 => Exists fun h => Exists fun x => S.P Z_1 …
    -/
  · rintro ⟨Z, g, h, H, mem⟩
    /-
      case mp.intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      Y Z✝ : C
      g✝ : Quiver.Hom Y Z✝
      Z : C
      g : Quiver.Hom Z✝ Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj Y)
      H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
      mem : S.P Z
      ⊢ Exists fun X => Exists fun f => Exists fun h => Exists fun x => S.P X
    -/
    exact ⟨_, _, _, inv_rot_of_distTriang _ H, S.shift _ (-1) mem⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      Y Z : C
      g : Quiver.Hom Y Z
      ⊢ (Exists fun X => Exists fun f => Exists fun h => Exists fun x => S.P X) → Ex …
    -/
  · rintro ⟨Z, g, h, H, mem⟩
    /-
      case mpr.intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      Y Z✝ : C
      g✝ : Quiver.Hom Y Z✝
      Z : C
      g : Quiver.Hom Z Y
      h : Quiver.Hom Z✝ ((CategoryTheory.shiftFunctor C 1).obj Z)
      H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
      mem : S.P Z
      ⊢ Exists fun Z => Exists fun g => Exists fun h => Exists fun x => S.P Z
    -/
    exact ⟨_, _, _, rot_of_distTriang _ H, S.shift _ 1 mem⟩
    /-
      🎉 no goals
    -/


lemma W.mk {T : Triangle C} (hT : T ∈ distTriang C) (h : S.P T.obj₃) : S.W T.mor₁ :=
  ⟨_, _, _, hT, h⟩


lemma W.mk' {T : Triangle C} (hT : T ∈ distTriang C) (h : S.P T.obj₁) : S.W T.mor₂ := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : CategoryTheory.HasShift C Int
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    h : S.P T.obj₁
    ⊢ S.W T.mor₂
  -/
  rw [W_iff']
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : CategoryTheory.HasShift C Int
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    h : S.P T.obj₁
    ⊢ Exists fun X => Exists fun f => Exists fun h => Exists fun x => S.P X
  -/
  exact ⟨_, _, _, hT, h⟩
  /-
    🎉 no goals
  -/


lemma isoClosure_W : S.isoClosure.W = S.W := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : CategoryTheory.HasShift C Int
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    ⊢ Eq S.isoClosure.W S.W
  -/
  ext X Y f
  /-
    case h
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : CategoryTheory.HasShift C Int
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (S.isoClosure.W f) (S.W f)
  -/
  constructor
    /-
      case h.mp
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      X Y : C
      f : Quiver.Hom X Y
      ⊢ S.isoClosure.W f → S.W f
    -/
  · rintro ⟨Z, g, h, mem, ⟨Z', hZ', ⟨e⟩⟩⟩
    /-
      case h.mp.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      g : Quiver.Hom Y Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
      mem : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      Z' : C
      hZ' : S.P Z'
      e : CategoryTheory.Iso Z Z'
      ⊢ S.W f
    -/
    refine ⟨Z', g ≫ e.hom, e.inv ≫ h, isomorphic_distinguished _ mem _ ?_, hZ'⟩
    /-
      case h.mp.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      g : Quiver.Hom Y Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
      mem : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      Z' : C
      hZ' : S.P Z'
      e : CategoryTheory.Iso Z Z'
      ⊢ CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk f (CategoryTh …
    -/
    exact Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _) e.symm
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      X Y : C
      f : Quiver.Hom X Y
      ⊢ S.W f → S.isoClosure.W f
    -/
  · rintro ⟨Z, g, h, mem, hZ⟩
    /-
      case h.mpr.intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      g : Quiver.Hom Y Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
      mem : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      hZ : S.P Z
      ⊢ S.isoClosure.W f
    -/
    exact ⟨Z, g, h, mem, le_isoClosure _ _ hZ⟩
    /-
      🎉 no goals
    -/


instance respectsIso_W : S.W.RespectsIso where
  precomp {X' X Y} e (he : IsIso e) := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      X' X Y : C
      e : Quiver.Hom X' X
      he : CategoryTheory.IsIso e
      ⊢ ∀ (f : Quiver.Hom X Y), S.W f → S.W (CategoryTheory.CategoryStruct.comp e f)
    -/
    rintro f ⟨Z, g, h, mem, mem'⟩
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      X' X Y : C
      e : Quiver.Hom X' X
      he : CategoryTheory.IsIso e
      f : Quiver.Hom X Y
      Z : C
      g : Quiver.Hom Y Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
      mem : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      mem' : S.P Z
      ⊢ S.W (CategoryTheory.CategoryStruct.comp e f)
    -/
    refine ⟨Z, g, h ≫ inv e⟦(1 : ℤ)⟧', isomorphic_distinguished _ mem _ ?_, mem'⟩
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      X' X Y : C
      e : Quiver.Hom X' X
      he : CategoryTheory.IsIso e
      f : Quiver.Hom X Y
      Z : C
      g : Quiver.Hom Y Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
      mem : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      mem' : S.P Z
      ⊢ CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk (CategoryTheo …
    -/
    refine Triangle.isoMk _ _ (asIso e) (Iso.refl _) (Iso.refl _) (by aesop_cat) (by aesop_cat) ?_
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      X' X Y : C
      e : Quiver.Hom X' X
      he : CategoryTheory.IsIso e
      f : Quiver.Hom X Y
      Z : C
      g : Quiver.Hom Y Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
      mem : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      mem' : S.P Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.Trian …
    -/
    dsimp
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      X' X Y : C
      e : Quiver.Hom X' X
      he : CategoryTheory.IsIso e
      f : Quiver.Hom X Y
      Z : C
      g : Quiver.Hom Y Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
      mem : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      mem' : S.P Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp h …
    -/
    simp only [Functor.map_inv, assoc, IsIso.inv_hom_id, comp_id, id_comp]
    /-
      🎉 no goals
    -/
  postcomp {X Y Y'} e (he : IsIso e) := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      X Y Y' : C
      e : Quiver.Hom Y Y'
      he : CategoryTheory.IsIso e
      ⊢ ∀ (f : Quiver.Hom X Y), S.W f → S.W (CategoryTheory.CategoryStruct.comp f e)
    -/
    rintro f ⟨Z, g, h, mem, mem'⟩
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      X Y Y' : C
      e : Quiver.Hom Y Y'
      he : CategoryTheory.IsIso e
      f : Quiver.Hom X Y
      Z : C
      g : Quiver.Hom Y Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
      mem : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      mem' : S.P Z
      ⊢ S.W (CategoryTheory.CategoryStruct.comp f e)
    -/
    refine ⟨Z, inv e ≫ g, h, isomorphic_distinguished _ mem _ ?_, mem'⟩
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      X Y Y' : C
      e : Quiver.Hom Y Y'
      he : CategoryTheory.IsIso e
      f : Quiver.Hom X Y
      Z : C
      g : Quiver.Hom Y Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
      mem : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      mem' : S.P Z
      ⊢ CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk (CategoryTheo …
    -/
    exact Triangle.isoMk _ _ (Iso.refl _) (asIso e).symm (Iso.refl _)
    /-
      🎉 no goals
    -/


instance : S.W.ContainsIdentities := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : CategoryTheory.HasShift C Int
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    ⊢ S.W.ContainsIdentities
  -/
  rw [← isoClosure_W]
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : CategoryTheory.HasShift C Int
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    ⊢ S.isoClosure.W.ContainsIdentities
  -/
  exact ⟨fun X => ⟨_, _, _, contractible_distinguished X, zero _⟩⟩
  /-
    🎉 no goals
  -/


lemma W_of_isIso {X Y : C} (f : X ⟶ Y) [IsIso f] : S.W f := by
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝¹ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    ⊢ S.W f
  -/
  refine (S.W.arrow_mk_iso_iff ?_).1 (MorphismProperty.id_mem _ X)
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝¹ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    ⊢ CategoryTheory.Iso (CategoryTheory.Arrow.mk (CategoryTheory.CategoryStruct.i …
  -/
  exact Arrow.isoMk (Iso.refl _) (asIso f)
  /-
    🎉 no goals
  -/


lemma smul_mem_W_iff {X Y : C} (f : X ⟶ Y) (n : ℤˣ) :
    S.W (n • f) ↔ S.W f :=
                        /-
                          C : Type u_1
                          inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
                          inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
                          inst✝³ : CategoryTheory.HasShift C Int
                          inst✝² : CategoryTheory.Preadditive C
                          inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
                          inst✝ : CategoryTheory.Pretriangulated C
                          S : CategoryTheory.Triangulated.Subcategory C
                          X Y : C
                          f : Quiver.Hom X Y
                          n : Units Int
                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul n (CategoryTheory.Iso.re …
                        -/
  S.W.arrow_mk_iso_iff (Arrow.isoMk (n • (Iso.refl _)) (Iso.refl _))
                        /-
                          🎉 no goals
                        -/


lemma W.shift {X₁ X₂ : C} {f : X₁ ⟶ X₂} (hf : S.W f) (n : ℤ) : S.W (f⟦n⟧') := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : CategoryTheory.HasShift C Int
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    X₁ X₂ : C
    f : Quiver.Hom X₁ X₂
    hf : S.W f
    n : Int
    ⊢ S.W ((CategoryTheory.shiftFunctor C n).map f)
  -/
  rw [← smul_mem_W_iff _ _ (n.negOnePow)]
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : CategoryTheory.HasShift C Int
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    X₁ X₂ : C
    f : Quiver.Hom X₁ X₂
    hf : S.W f
    n : Int
    ⊢ S.W (HSMul.hSMul n.negOnePow ((CategoryTheory.shiftFunctor C n).map f))
  -/
  obtain ⟨X₃, g, h, hT, mem⟩ := hf
  /-
    case intro.intro.intro.intro
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : CategoryTheory.HasShift C Int
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    X₁ X₂ : C
    f : Quiver.Hom X₁ X₂
    n : Int
    X₃ : C
    g : Quiver.Hom X₂ X₃
    h : Quiver.Hom X₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cat …
    mem : S.P X₃
    ⊢ S.W (HSMul.hSMul n.negOnePow ((CategoryTheory.shiftFunctor C n).map f))
  -/
  exact ⟨_, _, _, Pretriangulated.Triangle.shift_distinguished _ hT n, S.shift _ _ mem⟩
  /-
    🎉 no goals
  -/


lemma W.unshift {X₁ X₂ : C} {f : X₁ ⟶ X₂} {n : ℤ} (hf : S.W (f⟦n⟧')) : S.W f :=
  (S.W.arrow_mk_iso_iff
     (Arrow.isoOfNatIso (shiftEquiv C n).unitIso (Arrow.mk f))).2 (hf.shift (-n))


instance : S.W.IsCompatibleWithShift ℤ where
  condition n := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      n : Int
      ⊢ Eq (S.W.inverseImage (CategoryTheory.shiftFunctor C n)) S.W
    -/
    ext K L f
    /-
      case h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      n : Int
      K L : C
      f : Quiver.Hom K L
      ⊢ Iff (S.W.inverseImage (CategoryTheory.shiftFunctor C n) f) (S.W f)
    -/
    exact ⟨fun hf => hf.unshift, fun hf => hf.shift n⟩
    /-
      🎉 no goals
    -/


instance [IsTriangulated C] : S.W.IsMultiplicative where
  comp_mem := by
    /-
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      ⊢ ∀ {X Y Z : C} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), S.W f → S.W g → S.W …
    -/
    rw [← isoClosure_W]
    /-
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      ⊢ ∀ {X Y Z : C} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), S.isoClosure.W f →  …
    -/
    rintro X₁ X₂ X₃ u₁₂ u₂₃ ⟨Z₁₂, v₁₂, w₁₂, H₁₂, mem₁₂⟩ ⟨Z₂₃, v₂₃, w₂₃, H₂₃, mem₂₃⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      X₁ X₂ X₃ : C
      u₁₂ : Quiver.Hom X₁ X₂
      u₂₃ : Quiver.Hom X₂ X₃
      Z₁₂ : C
      v₁₂ : Quiver.Hom X₂ Z₁₂
      w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      H₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      mem₁₂ : S.isoClosure.P Z₁₂
      Z₂₃ : C
      v₂₃ : Quiver.Hom X₃ Z₂₃
      w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
      H₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      mem₂₃ : S.isoClosure.P Z₂₃
      ⊢ S.isoClosure.W (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃)
    -/
    obtain ⟨Z₁₃, v₁₃, w₁₂, H₁₃⟩ := distinguished_cocone_triangle (u₁₂ ≫ u₂₃)
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      X₁ X₂ X₃ : C
      u₁₂ : Quiver.Hom X₁ X₂
      u₂₃ : Quiver.Hom X₂ X₃
      Z₁₂ : C
      v₁₂ : Quiver.Hom X₂ Z₁₂
      w₁₂✝ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      H₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      mem₁₂ : S.isoClosure.P Z₁₂
      Z₂₃ : C
      v₂₃ : Quiver.Hom X₃ Z₂₃
      w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
      H₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      mem₂₃ : S.isoClosure.P Z₂₃
      Z₁₃ : C
      v₁₃ : Quiver.Hom X₃ Z₁₃
      w₁₂ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      H₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      ⊢ S.isoClosure.W (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃)
    -/
    exact ⟨_, _, _, H₁₃, S.isoClosure.ext₂ _ (someOctahedron rfl H₁₂ H₂₃ H₁₃).mem mem₁₂ mem₂₃⟩
    /-
      🎉 no goals
    -/


lemma mem_W_iff_of_distinguished
    [ClosedUnderIsomorphisms S.P] (T : Triangle C) (hT : T ∈ distTriang C) :
    S.W T.mor₁ ↔ S.P T.obj₃ := by
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝¹ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    inst✝ : CategoryTheory.ClosedUnderIsomorphisms S.P
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Iff (S.W T.mor₁) (S.P T.obj₃)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.ClosedUnderIsomorphisms S.P
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ S.W T.mor₁ → S.P T.obj₃
    -/
  · rintro ⟨Z, g, h, hT', mem⟩
    /-
      case mp.intro.intro.intro.intro
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.ClosedUnderIsomorphisms S.P
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      Z : C
      g : Quiver.Hom T.obj₂ Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj T.obj₁)
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      mem : S.P Z
      ⊢ S.P T.obj₃
    -/
    obtain ⟨e, _⟩ := exists_iso_of_arrow_iso _ _ hT' hT (Iso.refl _)
    /-
      case mp.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.ClosedUnderIsomorphisms S.P
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      Z : C
      g : Quiver.Hom T.obj₂ Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj T.obj₁)
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      mem : S.P Z
      e : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk T.mor₁ g h) T
      h✝ : And (Eq e.hom.hom₁ (CategoryTheory.Iso.refl (CategoryTheory.Arrow.mk (Cat …
      ⊢ S.P T.obj₃
    -/
    exact mem_of_iso S.P (Triangle.π₃.mapIso e) mem
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.ClosedUnderIsomorphisms S.P
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ S.P T.obj₃ → S.W T.mor₁
    -/
  · intro h
    /-
      case mpr
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.ClosedUnderIsomorphisms S.P
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h : S.P T.obj₃
      ⊢ S.W T.mor₁
    -/
    exact ⟨_, _, _, hT, h⟩
    /-
      🎉 no goals
    -/


instance [IsTriangulated C] : S.W.HasLeftCalculusOfFractions where
  exists_leftFraction X Y φ := by
    /-
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      X Y : C
      φ : S.W.RightFraction X Y
      ⊢ Exists fun ψ => Eq (CategoryTheory.CategoryStruct.comp φ.f ψ.s) (CategoryThe …
    -/
    obtain ⟨Z, f, g, H, mem⟩ := φ.hs
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      X Y : C
      φ : S.W.RightFraction X Y
      Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj φ.X')
      H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
      mem : S.P Z
      ⊢ Exists fun ψ => Eq (CategoryTheory.CategoryStruct.comp φ.f ψ.s) (CategoryThe …
    -/
    obtain ⟨Y', s', f', mem'⟩ := distinguished_cocone_triangle₂ (g ≫ φ.f⟦1⟧')
    obtain ⟨b, ⟨hb₁, _⟩⟩ :=
      complete_distinguished_triangle_morphism₂ _ _ H mem' φ.f (𝟙 Z) (by simp)
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      X Y : C
      φ : S.W.RightFraction X Y
      Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj φ.X')
      H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
      mem : S.P Z
      Y' : C
      s' : Quiver.Hom Y Y'
      f' : Quiver.Hom Y' Z
      mem' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      b : Quiver.Hom (CategoryTheory.Pretriangulated.Triangle.mk φ.s f g).obj₂ (Cate …
      hb₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.T …
      right✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulate …
      ⊢ Exists fun ψ => Eq (CategoryTheory.CategoryStruct.comp φ.f ψ.s) (CategoryThe …
    -/
    exact ⟨MorphismProperty.LeftFraction.mk b s' ⟨_, _, _, mem', mem⟩, hb₁.symm⟩
    /-
      🎉 no goals
    -/
  ext := by
    /-
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      ⊢ ∀ ⦃X' X Y : C⦄ (f₁ f₂ : Quiver.Hom X Y) (s : Quiver.Hom X' X), S.W s → Eq (C …
    -/
    rintro X' X Y f₁ f₂ s ⟨Z, g, h, H, mem⟩ hf₁
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      X' X Y : C
      f₁ f₂ : Quiver.Hom X Y
      s : Quiver.Hom X' X
      Z : C
      g : Quiver.Hom X Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X')
      H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
      mem : S.P Z
      hf₁ : Eq (CategoryTheory.CategoryStruct.comp s f₁) (CategoryTheory.CategoryStr …
      ⊢ Exists fun Y' => Exists fun t => Exists fun x => Eq (CategoryTheory.Category …
    -/
    have hf₂ : s ≫ (f₁ - f₂) = 0 := by rw [comp_sub, hf₁, sub_self]
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      X' X Y : C
      f₁ f₂ : Quiver.Hom X Y
      s : Quiver.Hom X' X
      Z : C
      g : Quiver.Hom X Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X')
      H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
      mem : S.P Z
      hf₁ : Eq (CategoryTheory.CategoryStruct.comp s f₁) (CategoryTheory.CategoryStr …
      hf₂ : Eq (CategoryTheory.CategoryStruct.comp s (HSub.hSub f₁ f₂)) 0
      ⊢ Exists fun Y' => Exists fun t => Exists fun x => Eq (CategoryTheory.Category …
    -/
    obtain ⟨q, hq⟩ := Triangle.yoneda_exact₂ _ H _ hf₂
    /-
      case intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      X' X Y : C
      f₁ f₂ : Quiver.Hom X Y
      s : Quiver.Hom X' X
      Z : C
      g : Quiver.Hom X Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X')
      H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
      mem : S.P Z
      hf₁ : Eq (CategoryTheory.CategoryStruct.comp s f₁) (CategoryTheory.CategoryStr …
      hf₂ : Eq (CategoryTheory.CategoryStruct.comp s (HSub.hSub f₁ f₂)) 0
      q : Quiver.Hom (CategoryTheory.Pretriangulated.Triangle.mk s g h).obj₃ Y
      hq : Eq (HSub.hSub f₁ f₂) (CategoryTheory.CategoryStruct.comp (CategoryTheory. …
      ⊢ Exists fun Y' => Exists fun t => Exists fun x => Eq (CategoryTheory.Category …
    -/
    obtain ⟨Y', r, t, mem'⟩ := distinguished_cocone_triangle q
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      X' X Y : C
      f₁ f₂ : Quiver.Hom X Y
      s : Quiver.Hom X' X
      Z : C
      g : Quiver.Hom X Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X')
      H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
      mem : S.P Z
      hf₁ : Eq (CategoryTheory.CategoryStruct.comp s f₁) (CategoryTheory.CategoryStr …
      hf₂ : Eq (CategoryTheory.CategoryStruct.comp s (HSub.hSub f₁ f₂)) 0
      q : Quiver.Hom (CategoryTheory.Pretriangulated.Triangle.mk s g h).obj₃ Y
      hq : Eq (HSub.hSub f₁ f₂) (CategoryTheory.CategoryStruct.comp (CategoryTheory. …
      Y' : C
      r : Quiver.Hom Y Y'
      t : Quiver.Hom Y' ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Pretr …
      mem' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      ⊢ Exists fun Y' => Exists fun t => Exists fun x => Eq (CategoryTheory.Category …
    -/
    refine ⟨Y', r, ?_, ?_⟩
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.refine_1
        C : Type u_1
        inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
        inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
        inst✝⁴ : CategoryTheory.HasShift C Int
        inst✝³ : CategoryTheory.Preadditive C
        inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
        inst✝¹ : CategoryTheory.Pretriangulated C
        S : CategoryTheory.Triangulated.Subcategory C
        inst✝ : CategoryTheory.IsTriangulated C
        X' X Y : C
        f₁ f₂ : Quiver.Hom X Y
        s : Quiver.Hom X' X
        Z : C
        g : Quiver.Hom X Z
        h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X')
        H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
        mem : S.P Z
        hf₁ : Eq (CategoryTheory.CategoryStruct.comp s f₁) (CategoryTheory.CategoryStr …
        hf₂ : Eq (CategoryTheory.CategoryStruct.comp s (HSub.hSub f₁ f₂)) 0
        q : Quiver.Hom (CategoryTheory.Pretriangulated.Triangle.mk s g h).obj₃ Y
        hq : Eq (HSub.hSub f₁ f₂) (CategoryTheory.CategoryStruct.comp (CategoryTheory. …
        Y' : C
        r : Quiver.Hom Y Y'
        t : Quiver.Hom Y' ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Pretr …
        mem' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
        ⊢ S.W r
      -/
    · exact ⟨_, _, _, rot_of_distTriang _ mem', S.shift _ _ mem⟩
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.refine_2
        C : Type u_1
        inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
        inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
        inst✝⁴ : CategoryTheory.HasShift C Int
        inst✝³ : CategoryTheory.Preadditive C
        inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
        inst✝¹ : CategoryTheory.Pretriangulated C
        S : CategoryTheory.Triangulated.Subcategory C
        inst✝ : CategoryTheory.IsTriangulated C
        X' X Y : C
        f₁ f₂ : Quiver.Hom X Y
        s : Quiver.Hom X' X
        Z : C
        g : Quiver.Hom X Z
        h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X')
        H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
        mem : S.P Z
        hf₁ : Eq (CategoryTheory.CategoryStruct.comp s f₁) (CategoryTheory.CategoryStr …
        hf₂ : Eq (CategoryTheory.CategoryStruct.comp s (HSub.hSub f₁ f₂)) 0
        q : Quiver.Hom (CategoryTheory.Pretriangulated.Triangle.mk s g h).obj₃ Y
        hq : Eq (HSub.hSub f₁ f₂) (CategoryTheory.CategoryStruct.comp (CategoryTheory. …
        Y' : C
        r : Quiver.Hom Y Y'
        t : Quiver.Hom Y' ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Pretr …
        mem' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f₁ r) (CategoryTheory.CategoryStruct. …
      -/
    · have eq := comp_distTriang_mor_zero₁₂ _ mem'
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.refine_2
        C : Type u_1
        inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
        inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
        inst✝⁴ : CategoryTheory.HasShift C Int
        inst✝³ : CategoryTheory.Preadditive C
        inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
        inst✝¹ : CategoryTheory.Pretriangulated C
        S : CategoryTheory.Triangulated.Subcategory C
        inst✝ : CategoryTheory.IsTriangulated C
        X' X Y : C
        f₁ f₂ : Quiver.Hom X Y
        s : Quiver.Hom X' X
        Z : C
        g : Quiver.Hom X Z
        h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X')
        H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
        mem : S.P Z
        hf₁ : Eq (CategoryTheory.CategoryStruct.comp s f₁) (CategoryTheory.CategoryStr …
        hf₂ : Eq (CategoryTheory.CategoryStruct.comp s (HSub.hSub f₁ f₂)) 0
        q : Quiver.Hom (CategoryTheory.Pretriangulated.Triangle.mk s g h).obj₃ Y
        hq : Eq (HSub.hSub f₁ f₂) (CategoryTheory.CategoryStruct.comp (CategoryTheory. …
        Y' : C
        r : Quiver.Hom Y Y'
        t : Quiver.Hom Y' ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Pretr …
        mem' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
        eq : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.Tr …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f₁ r) (CategoryTheory.CategoryStruct. …
      -/
      dsimp at eq
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.refine_2
        C : Type u_1
        inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
        inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
        inst✝⁴ : CategoryTheory.HasShift C Int
        inst✝³ : CategoryTheory.Preadditive C
        inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
        inst✝¹ : CategoryTheory.Pretriangulated C
        S : CategoryTheory.Triangulated.Subcategory C
        inst✝ : CategoryTheory.IsTriangulated C
        X' X Y : C
        f₁ f₂ : Quiver.Hom X Y
        s : Quiver.Hom X' X
        Z : C
        g : Quiver.Hom X Z
        h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X')
        H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
        mem : S.P Z
        hf₁ : Eq (CategoryTheory.CategoryStruct.comp s f₁) (CategoryTheory.CategoryStr …
        hf₂ : Eq (CategoryTheory.CategoryStruct.comp s (HSub.hSub f₁ f₂)) 0
        q : Quiver.Hom (CategoryTheory.Pretriangulated.Triangle.mk s g h).obj₃ Y
        hq : Eq (HSub.hSub f₁ f₂) (CategoryTheory.CategoryStruct.comp (CategoryTheory. …
        Y' : C
        r : Quiver.Hom Y Y'
        t : Quiver.Hom Y' ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Pretr …
        mem' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
        eq : Eq (CategoryTheory.CategoryStruct.comp q r) 0
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f₁ r) (CategoryTheory.CategoryStruct. …
      -/
      rw [← sub_eq_zero, ← sub_comp, hq, assoc, eq, comp_zero]
      /-
        🎉 no goals
      -/


instance [IsTriangulated C] : S.W.HasRightCalculusOfFractions where
  exists_rightFraction X Y φ := by
    /-
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      X Y : C
      φ : S.W.LeftFraction X Y
      ⊢ Exists fun ψ => Eq (CategoryTheory.CategoryStruct.comp ψ.s φ.f) (CategoryThe …
    -/
    obtain ⟨Z, f, g, H, mem⟩ := φ.hs
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      X Y : C
      φ : S.W.LeftFraction X Y
      Z : C
      f : Quiver.Hom φ.Y' Z
      g : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj Y)
      H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
      mem : S.P Z
      ⊢ Exists fun ψ => Eq (CategoryTheory.CategoryStruct.comp ψ.s φ.f) (CategoryThe …
    -/
    obtain ⟨X', f', h', mem'⟩ := distinguished_cocone_triangle₁ (φ.f ≫ f)
    obtain ⟨a, ⟨ha₁, _⟩⟩ := complete_distinguished_triangle_morphism₁ _ _
      mem' H φ.f (𝟙 Z) (by simp)
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      X Y : C
      φ : S.W.LeftFraction X Y
      Z : C
      f : Quiver.Hom φ.Y' Z
      g : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj Y)
      H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
      mem : S.P Z
      X' : C
      f' : Quiver.Hom X' X
      h' : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X')
      mem' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      a : Quiver.Hom (CategoryTheory.Pretriangulated.Triangle.mk f' (CategoryTheory. …
      ha₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.T …
      right✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulate …
      ⊢ Exists fun ψ => Eq (CategoryTheory.CategoryStruct.comp ψ.s φ.f) (CategoryThe …
    -/
    exact ⟨MorphismProperty.RightFraction.mk f' ⟨_, _, _, mem', mem⟩ a, ha₁⟩
    /-
      🎉 no goals
    -/
  ext Y Z Z' f₁ f₂ s hs hf₁ := by
    /-
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      Y Z Z' : C
      f₁ f₂ : Quiver.Hom Y Z
      s : Quiver.Hom Z Z'
      hs : S.W s
      hf₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ s) (CategoryTheory.CategoryStr …
      ⊢ Exists fun X' => Exists fun t => Exists fun x => Eq (CategoryTheory.Category …
    -/
    rw [S.W_iff'] at hs
    /-
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      Y Z Z' : C
      f₁ f₂ : Quiver.Hom Y Z
      s : Quiver.Hom Z Z'
      hs : Exists fun X => Exists fun f => Exists fun h => Exists fun x => S.P X
      hf₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ s) (CategoryTheory.CategoryStr …
      ⊢ Exists fun X' => Exists fun t => Exists fun x => Eq (CategoryTheory.Category …
    -/
    obtain ⟨Z, g, h, H, mem⟩ := hs
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      Y Z✝ Z' : C
      f₁ f₂ : Quiver.Hom Y Z✝
      s : Quiver.Hom Z✝ Z'
      hf₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ s) (CategoryTheory.CategoryStr …
      Z : C
      g : Quiver.Hom Z Z✝
      h : Quiver.Hom Z' ((CategoryTheory.shiftFunctor C 1).obj Z)
      H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
      mem : S.P Z
      ⊢ Exists fun X' => Exists fun t => Exists fun x => Eq (CategoryTheory.Category …
    -/
    have hf₂ : (f₁ - f₂) ≫ s = 0 := by rw [sub_comp, hf₁, sub_self]
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      Y Z✝ Z' : C
      f₁ f₂ : Quiver.Hom Y Z✝
      s : Quiver.Hom Z✝ Z'
      hf₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ s) (CategoryTheory.CategoryStr …
      Z : C
      g : Quiver.Hom Z Z✝
      h : Quiver.Hom Z' ((CategoryTheory.shiftFunctor C 1).obj Z)
      H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
      mem : S.P Z
      hf₂ : Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub f₁ f₂) s) 0
      ⊢ Exists fun X' => Exists fun t => Exists fun x => Eq (CategoryTheory.Category …
    -/
    obtain ⟨q, hq⟩ := Triangle.coyoneda_exact₂ _ H _ hf₂
    /-
      case intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      Y Z✝ Z' : C
      f₁ f₂ : Quiver.Hom Y Z✝
      s : Quiver.Hom Z✝ Z'
      hf₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ s) (CategoryTheory.CategoryStr …
      Z : C
      g : Quiver.Hom Z Z✝
      h : Quiver.Hom Z' ((CategoryTheory.shiftFunctor C 1).obj Z)
      H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
      mem : S.P Z
      hf₂ : Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub f₁ f₂) s) 0
      q : Quiver.Hom Y (CategoryTheory.Pretriangulated.Triangle.mk g s h).obj₁
      hq : Eq (HSub.hSub f₁ f₂) (CategoryTheory.CategoryStruct.comp q (CategoryTheor …
      ⊢ Exists fun X' => Exists fun t => Exists fun x => Eq (CategoryTheory.Category …
    -/
    obtain ⟨Y', r, t, mem'⟩ := distinguished_cocone_triangle₁ q
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated C
      S : CategoryTheory.Triangulated.Subcategory C
      inst✝ : CategoryTheory.IsTriangulated C
      Y Z✝ Z' : C
      f₁ f₂ : Quiver.Hom Y Z✝
      s : Quiver.Hom Z✝ Z'
      hf₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ s) (CategoryTheory.CategoryStr …
      Z : C
      g : Quiver.Hom Z Z✝
      h : Quiver.Hom Z' ((CategoryTheory.shiftFunctor C 1).obj Z)
      H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
      mem : S.P Z
      hf₂ : Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub f₁ f₂) s) 0
      q : Quiver.Hom Y (CategoryTheory.Pretriangulated.Triangle.mk g s h).obj₁
      hq : Eq (HSub.hSub f₁ f₂) (CategoryTheory.CategoryStruct.comp q (CategoryTheor …
      Y' : C
      r : Quiver.Hom Y' Y
      t : Quiver.Hom (CategoryTheory.Pretriangulated.Triangle.mk g s h).obj₁ ((Categ …
      mem' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      ⊢ Exists fun X' => Exists fun t => Exists fun x => Eq (CategoryTheory.Category …
    -/
    refine ⟨Y', r, ?_, ?_⟩
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.refine_1
        C : Type u_1
        inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
        inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
        inst✝⁴ : CategoryTheory.HasShift C Int
        inst✝³ : CategoryTheory.Preadditive C
        inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
        inst✝¹ : CategoryTheory.Pretriangulated C
        S : CategoryTheory.Triangulated.Subcategory C
        inst✝ : CategoryTheory.IsTriangulated C
        Y Z✝ Z' : C
        f₁ f₂ : Quiver.Hom Y Z✝
        s : Quiver.Hom Z✝ Z'
        hf₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ s) (CategoryTheory.CategoryStr …
        Z : C
        g : Quiver.Hom Z Z✝
        h : Quiver.Hom Z' ((CategoryTheory.shiftFunctor C 1).obj Z)
        H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
        mem : S.P Z
        hf₂ : Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub f₁ f₂) s) 0
        q : Quiver.Hom Y (CategoryTheory.Pretriangulated.Triangle.mk g s h).obj₁
        hq : Eq (HSub.hSub f₁ f₂) (CategoryTheory.CategoryStruct.comp q (CategoryTheor …
        Y' : C
        r : Quiver.Hom Y' Y
        t : Quiver.Hom (CategoryTheory.Pretriangulated.Triangle.mk g s h).obj₁ ((Categ …
        mem' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
        ⊢ S.W r
      -/
    · exact ⟨_, _, _, mem', mem⟩
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.refine_2
        C : Type u_1
        inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
        inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
        inst✝⁴ : CategoryTheory.HasShift C Int
        inst✝³ : CategoryTheory.Preadditive C
        inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
        inst✝¹ : CategoryTheory.Pretriangulated C
        S : CategoryTheory.Triangulated.Subcategory C
        inst✝ : CategoryTheory.IsTriangulated C
        Y Z✝ Z' : C
        f₁ f₂ : Quiver.Hom Y Z✝
        s : Quiver.Hom Z✝ Z'
        hf₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ s) (CategoryTheory.CategoryStr …
        Z : C
        g : Quiver.Hom Z Z✝
        h : Quiver.Hom Z' ((CategoryTheory.shiftFunctor C 1).obj Z)
        H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
        mem : S.P Z
        hf₂ : Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub f₁ f₂) s) 0
        q : Quiver.Hom Y (CategoryTheory.Pretriangulated.Triangle.mk g s h).obj₁
        hq : Eq (HSub.hSub f₁ f₂) (CategoryTheory.CategoryStruct.comp q (CategoryTheor …
        Y' : C
        r : Quiver.Hom Y' Y
        t : Quiver.Hom (CategoryTheory.Pretriangulated.Triangle.mk g s h).obj₁ ((Categ …
        mem' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp r f₁) (CategoryTheory.CategoryStruct. …
      -/
    · have eq := comp_distTriang_mor_zero₁₂ _ mem'
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.refine_2
        C : Type u_1
        inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
        inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
        inst✝⁴ : CategoryTheory.HasShift C Int
        inst✝³ : CategoryTheory.Preadditive C
        inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
        inst✝¹ : CategoryTheory.Pretriangulated C
        S : CategoryTheory.Triangulated.Subcategory C
        inst✝ : CategoryTheory.IsTriangulated C
        Y Z✝ Z' : C
        f₁ f₂ : Quiver.Hom Y Z✝
        s : Quiver.Hom Z✝ Z'
        hf₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ s) (CategoryTheory.CategoryStr …
        Z : C
        g : Quiver.Hom Z Z✝
        h : Quiver.Hom Z' ((CategoryTheory.shiftFunctor C 1).obj Z)
        H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
        mem : S.P Z
        hf₂ : Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub f₁ f₂) s) 0
        q : Quiver.Hom Y (CategoryTheory.Pretriangulated.Triangle.mk g s h).obj₁
        hq : Eq (HSub.hSub f₁ f₂) (CategoryTheory.CategoryStruct.comp q (CategoryTheor …
        Y' : C
        r : Quiver.Hom Y' Y
        t : Quiver.Hom (CategoryTheory.Pretriangulated.Triangle.mk g s h).obj₁ ((Categ …
        mem' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
        eq : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.Tr …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp r f₁) (CategoryTheory.CategoryStruct. …
      -/
      dsimp at eq
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.refine_2
        C : Type u_1
        inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
        inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
        inst✝⁴ : CategoryTheory.HasShift C Int
        inst✝³ : CategoryTheory.Preadditive C
        inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
        inst✝¹ : CategoryTheory.Pretriangulated C
        S : CategoryTheory.Triangulated.Subcategory C
        inst✝ : CategoryTheory.IsTriangulated C
        Y Z✝ Z' : C
        f₁ f₂ : Quiver.Hom Y Z✝
        s : Quiver.Hom Z✝ Z'
        hf₁ : Eq (CategoryTheory.CategoryStruct.comp f₁ s) (CategoryTheory.CategoryStr …
        Z : C
        g : Quiver.Hom Z Z✝
        h : Quiver.Hom Z' ((CategoryTheory.shiftFunctor C 1).obj Z)
        H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
        mem : S.P Z
        hf₂ : Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub f₁ f₂) s) 0
        q : Quiver.Hom Y (CategoryTheory.Pretriangulated.Triangle.mk g s h).obj₁
        hq : Eq (HSub.hSub f₁ f₂) (CategoryTheory.CategoryStruct.comp q (CategoryTheor …
        Y' : C
        r : Quiver.Hom Y' Y
        t : Quiver.Hom (CategoryTheory.Pretriangulated.Triangle.mk g s h).obj₁ ((Categ …
        mem' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
        eq : Eq (CategoryTheory.CategoryStruct.comp r q) 0
        ⊢ Eq (CategoryTheory.CategoryStruct.comp r f₁) (CategoryTheory.CategoryStruct. …
      -/
      rw [← sub_eq_zero, ← comp_sub, hq, reassoc_of% eq, zero_comp]
      /-
        🎉 no goals
      -/


instance [IsTriangulated C] : S.W.IsCompatibleWithTriangulation := ⟨by
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝¹ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    inst✝ : CategoryTheory.IsTriangulated C
    ⊢ ∀ (T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Catego …
  -/
  rintro T₁ T₃ mem₁ mem₃ a b ⟨Z₅, g₅, h₅, mem₅, mem₅'⟩ ⟨Z₄, g₄, h₄, mem₄, mem₄'⟩ comm
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝¹ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    inst✝ : CategoryTheory.IsTriangulated C
    T₁ T₃ : CategoryTheory.Pretriangulated.Triangle C
    mem₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    mem₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₃
    a : Quiver.Hom T₁.obj₁ T₃.obj₁
    b : Quiver.Hom T₁.obj₂ T₃.obj₂
    Z₅ : C
    g₅ : Quiver.Hom T₃.obj₁ Z₅
    h₅ : Quiver.Hom Z₅ ((CategoryTheory.shiftFunctor C 1).obj T₁.obj₁)
    mem₅ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    mem₅' : S.P Z₅
    Z₄ : C
    g₄ : Quiver.Hom T₃.obj₂ Z₄
    h₄ : Quiver.Hom Z₄ ((CategoryTheory.shiftFunctor C 1).obj T₁.obj₂)
    mem₄ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    mem₄' : S.P Z₄
    comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Categ …
    ⊢ Exists fun c => Exists fun x => And (Eq (CategoryTheory.CategoryStruct.comp  …
  -/
  obtain ⟨Z₂, g₂, h₂, mem₂⟩ := distinguished_cocone_triangle (T₁.mor₁ ≫ b)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝¹ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    inst✝ : CategoryTheory.IsTriangulated C
    T₁ T₃ : CategoryTheory.Pretriangulated.Triangle C
    mem₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    mem₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₃
    a : Quiver.Hom T₁.obj₁ T₃.obj₁
    b : Quiver.Hom T₁.obj₂ T₃.obj₂
    Z₅ : C
    g₅ : Quiver.Hom T₃.obj₁ Z₅
    h₅ : Quiver.Hom Z₅ ((CategoryTheory.shiftFunctor C 1).obj T₁.obj₁)
    mem₅ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    mem₅' : S.P Z₅
    Z₄ : C
    g₄ : Quiver.Hom T₃.obj₂ Z₄
    h₄ : Quiver.Hom Z₄ ((CategoryTheory.shiftFunctor C 1).obj T₁.obj₂)
    mem₄ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    mem₄' : S.P Z₄
    comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Categ …
    Z₂ : C
    g₂ : Quiver.Hom T₃.obj₂ Z₂
    h₂ : Quiver.Hom Z₂ ((CategoryTheory.shiftFunctor C 1).obj T₁.obj₁)
    mem₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    ⊢ Exists fun c => Exists fun x => And (Eq (CategoryTheory.CategoryStruct.comp  …
  -/
  have H := someOctahedron rfl mem₁ mem₄ mem₂
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝¹ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    inst✝ : CategoryTheory.IsTriangulated C
    T₁ T₃ : CategoryTheory.Pretriangulated.Triangle C
    mem₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    mem₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₃
    a : Quiver.Hom T₁.obj₁ T₃.obj₁
    b : Quiver.Hom T₁.obj₂ T₃.obj₂
    Z₅ : C
    g₅ : Quiver.Hom T₃.obj₁ Z₅
    h₅ : Quiver.Hom Z₅ ((CategoryTheory.shiftFunctor C 1).obj T₁.obj₁)
    mem₅ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    mem₅' : S.P Z₅
    Z₄ : C
    g₄ : Quiver.Hom T₃.obj₂ Z₄
    h₄ : Quiver.Hom Z₄ ((CategoryTheory.shiftFunctor C 1).obj T₁.obj₂)
    mem₄ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    mem₄' : S.P Z₄
    comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Categ …
    Z₂ : C
    g₂ : Quiver.Hom T₃.obj₂ Z₂
    h₂ : Quiver.Hom Z₂ ((CategoryTheory.shiftFunctor C 1).obj T₁.obj₁)
    mem₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    H : CategoryTheory.Triangulated.Octahedron ⋯ mem₁ mem₄ mem₂
    ⊢ Exists fun c => Exists fun x => And (Eq (CategoryTheory.CategoryStruct.comp  …
  -/
  have H' := someOctahedron comm.symm mem₅ mem₃ mem₂
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝¹ : CategoryTheory.Pretriangulated C
    S : CategoryTheory.Triangulated.Subcategory C
    inst✝ : CategoryTheory.IsTriangulated C
    T₁ T₃ : CategoryTheory.Pretriangulated.Triangle C
    mem₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    mem₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₃
    a : Quiver.Hom T₁.obj₁ T₃.obj₁
    b : Quiver.Hom T₁.obj₂ T₃.obj₂
    Z₅ : C
    g₅ : Quiver.Hom T₃.obj₁ Z₅
    h₅ : Quiver.Hom Z₅ ((CategoryTheory.shiftFunctor C 1).obj T₁.obj₁)
    mem₅ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    mem₅' : S.P Z₅
    Z₄ : C
    g₄ : Quiver.Hom T₃.obj₂ Z₄
    h₄ : Quiver.Hom Z₄ ((CategoryTheory.shiftFunctor C 1).obj T₁.obj₂)
    mem₄ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    mem₄' : S.P Z₄
    comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Categ …
    Z₂ : C
    g₂ : Quiver.Hom T₃.obj₂ Z₂
    h₂ : Quiver.Hom Z₂ ((CategoryTheory.shiftFunctor C 1).obj T₁.obj₁)
    mem₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    H : CategoryTheory.Triangulated.Octahedron ⋯ mem₁ mem₄ mem₂
    H' : CategoryTheory.Triangulated.Octahedron ⋯ mem₅ mem₃ mem₂
    ⊢ Exists fun c => Exists fun x => And (Eq (CategoryTheory.CategoryStruct.comp  …
  -/
  let φ : T₁ ⟶ T₃ := H.triangleMorphism₁ ≫ H'.triangleMorphism₂
  exact ⟨φ.hom₃, S.W.comp_mem _ _ (W.mk S H.mem mem₄') (W.mk' S H'.mem mem₅'),
    by simpa [φ] using φ.comm₂, by simpa [φ] using φ.comm₃⟩⟩


lemma ext₁ [ClosedUnderIsomorphisms S.P] (h₂ : S.P T.obj₂) (h₃ : S.P T.obj₃) :
    S.P T.obj₁ :=
  S.ext₂ _ (inv_rot_of_distTriang _ hT) (S.shift _ _ h₃) h₂


lemma ext₃ [ClosedUnderIsomorphisms S.P] (h₁ : S.P T.obj₁) (h₂ : S.P T.obj₂) :
    S.P T.obj₃ :=
  S.ext₂ _ (rot_of_distTriang _ hT) h₂ (S.shift _ _ h₁)


lemma ext₁' (h₂ : S.P T.obj₂) (h₃ : S.P T.obj₃) :
    CategoryTheory.isoClosure S.P T.obj₁ :=
  S.ext₂' _ (inv_rot_of_distTriang _ hT) (S.shift _ _ h₃) h₂


lemma ext₃' (h₁ : S.P T.obj₁) (h₂ : S.P T.obj₂) :
    CategoryTheory.isoClosure S.P T.obj₃ :=
  S.ext₂' _ (rot_of_distTriang _ hT) h₂ (S.shift _ _ h₁)



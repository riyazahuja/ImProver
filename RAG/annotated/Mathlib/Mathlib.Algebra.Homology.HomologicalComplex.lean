/-- A `HomologicalComplex V c` with a "shape" controlled by `c : ComplexShape ι`
has chain groups `X i` (objects in `V`) indexed by `i : ι`,
and a differential `d i j` whenever `c.Rel i j`.

We in fact ask for differentials `d i j` for all `i j : ι`,
but have a field `shape` requiring that these are zero when not allowed by `c`.
This avoids a lot of dependent type theory hell!

The composite of any two differentials `d i j ≫ d j k` must be zero.
-/
structure HomologicalComplex (c : ComplexShape ι) where
  X : ι → V
  d : ∀ i j, X i ⟶ X j
  shape : ∀ i j, ¬c.Rel i j → d i j = 0 := by aesop_cat
  d_comp_d' : ∀ i j k, c.Rel i j → c.Rel j k → d i j ≫ d j k = 0 := by aesop_cat


@[reassoc (attr := simp)]
theorem d_comp_d (C : HomologicalComplex V c) (i j k : ι) : C.d i j ≫ C.d j k = 0 := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    i j k : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.d i j) (C.d j k)) 0
  -/
  by_cases hij : c.Rel i j
    /-
      case pos
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      c : ComplexShape ι
      C : HomologicalComplex V c
      i j k : ι
      hij : c.Rel i j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.d i j) (C.d j k)) 0
    -/
  · by_cases hjk : c.Rel j k
      /-
        case pos
        ι : Type u_1
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        c : ComplexShape ι
        C : HomologicalComplex V c
        i j k : ι
        hij : c.Rel i j
        hjk : c.Rel j k
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.d i j) (C.d j k)) 0
      -/
    · exact C.d_comp_d' i j k hij hjk
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u_1
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        c : ComplexShape ι
        C : HomologicalComplex V c
        i j k : ι
        hij : c.Rel i j
        hjk : Not (c.Rel j k)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.d i j) (C.d j k)) 0
      -/
    · rw [C.shape j k hjk, comp_zero]
      /-
        🎉 no goals
      -/
    /-
      case neg
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      c : ComplexShape ι
      C : HomologicalComplex V c
      i j k : ι
      hij : Not (c.Rel i j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.d i j) (C.d j k)) 0
    -/
  · rw [C.shape i j hij, zero_comp]
    /-
      🎉 no goals
    -/


theorem ext {C₁ C₂ : HomologicalComplex V c} (h_X : C₁.X = C₂.X)
    (h_d :
      ∀ i j : ι,
        c.Rel i j → C₁.d i j ≫ eqToHom (congr_fun h_X j) = eqToHom (congr_fun h_X i) ≫ C₂.d i j) :
    C₁ = C₂ := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C₁ C₂ : HomologicalComplex V c
    h_X : Eq C₁.X C₂.X
    h_d : ∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp (C₁.d i  …
    ⊢ Eq C₁ C₂
  -/
  obtain ⟨X₁, d₁, s₁, h₁⟩ := C₁
  /-
    case mk
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C₂ : HomologicalComplex V c
    X₁ : ι → V
    d₁ : (i j : ι) → Quiver.Hom (X₁ i) (X₁ j)
    s₁ : ∀ (i j : ι), Not (c.Rel i j) → Eq (d₁ i j) 0
    h₁ : ∀ (i j k : ι), c.Rel i j → c.Rel j k → Eq (CategoryTheory.CategoryStruct. …
    h_X : Eq { X := X₁, d := d₁, shape := s₁, d_comp_d' := h₁ }.X C₂.X
    h_d : ∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp ({ X :=  …
    ⊢ Eq { X := X₁, d := d₁, shape := s₁, d_comp_d' := h₁ } C₂
  -/
  obtain ⟨X₂, d₂, s₂, h₂⟩ := C₂
  /-
    case mk.mk
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    X₁ : ι → V
    d₁ : (i j : ι) → Quiver.Hom (X₁ i) (X₁ j)
    s₁ : ∀ (i j : ι), Not (c.Rel i j) → Eq (d₁ i j) 0
    h₁ : ∀ (i j k : ι), c.Rel i j → c.Rel j k → Eq (CategoryTheory.CategoryStruct. …
    X₂ : ι → V
    d₂ : (i j : ι) → Quiver.Hom (X₂ i) (X₂ j)
    s₂ : ∀ (i j : ι), Not (c.Rel i j) → Eq (d₂ i j) 0
    h₂ : ∀ (i j k : ι), c.Rel i j → c.Rel j k → Eq (CategoryTheory.CategoryStruct. …
    h_X : Eq { X := X₁, d := d₁, shape := s₁, d_comp_d' := h₁ }.X { X := X₂, d :=  …
    h_d : ∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp ({ X :=  …
    ⊢ Eq { X := X₁, d := d₁, shape := s₁, d_comp_d' := h₁ } { X := X₂, d := d₂, sh …
  -/
  dsimp at h_X
  /-
    case mk.mk
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    X₁ : ι → V
    d₁ : (i j : ι) → Quiver.Hom (X₁ i) (X₁ j)
    s₁ : ∀ (i j : ι), Not (c.Rel i j) → Eq (d₁ i j) 0
    h₁ : ∀ (i j k : ι), c.Rel i j → c.Rel j k → Eq (CategoryTheory.CategoryStruct. …
    X₂ : ι → V
    d₂ : (i j : ι) → Quiver.Hom (X₂ i) (X₂ j)
    s₂ : ∀ (i j : ι), Not (c.Rel i j) → Eq (d₂ i j) 0
    h₂ : ∀ (i j k : ι), c.Rel i j → c.Rel j k → Eq (CategoryTheory.CategoryStruct. …
    h_X : Eq X₁ X₂
    h_d : ∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp ({ X :=  …
    ⊢ Eq { X := X₁, d := d₁, shape := s₁, d_comp_d' := h₁ } { X := X₂, d := d₂, sh …
  -/
  subst h_X
  /-
    case mk.mk
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    X₁ : ι → V
    d₁ : (i j : ι) → Quiver.Hom (X₁ i) (X₁ j)
    s₁ : ∀ (i j : ι), Not (c.Rel i j) → Eq (d₁ i j) 0
    h₁ : ∀ (i j k : ι), c.Rel i j → c.Rel j k → Eq (CategoryTheory.CategoryStruct. …
    d₂ : (i j : ι) → Quiver.Hom (X₁ i) (X₁ j)
    s₂ : ∀ (i j : ι), Not (c.Rel i j) → Eq (d₂ i j) 0
    h₂ : ∀ (i j k : ι), c.Rel i j → c.Rel j k → Eq (CategoryTheory.CategoryStruct. …
    h_d : ∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp ({ X :=  …
    ⊢ Eq { X := X₁, d := d₁, shape := s₁, d_comp_d' := h₁ } { X := X₁, d := d₂, sh …
  -/
  simp only [mk.injEq, heq_eq_eq, true_and]
  /-
    case mk.mk
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    X₁ : ι → V
    d₁ : (i j : ι) → Quiver.Hom (X₁ i) (X₁ j)
    s₁ : ∀ (i j : ι), Not (c.Rel i j) → Eq (d₁ i j) 0
    h₁ : ∀ (i j k : ι), c.Rel i j → c.Rel j k → Eq (CategoryTheory.CategoryStruct. …
    d₂ : (i j : ι) → Quiver.Hom (X₁ i) (X₁ j)
    s₂ : ∀ (i j : ι), Not (c.Rel i j) → Eq (d₂ i j) 0
    h₂ : ∀ (i j k : ι), c.Rel i j → c.Rel j k → Eq (CategoryTheory.CategoryStruct. …
    h_d : ∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp ({ X :=  …
    ⊢ Eq d₁ d₂
  -/
  ext i j
  /-
    case mk.mk.h.h
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    X₁ : ι → V
    d₁ : (i j : ι) → Quiver.Hom (X₁ i) (X₁ j)
    s₁ : ∀ (i j : ι), Not (c.Rel i j) → Eq (d₁ i j) 0
    h₁ : ∀ (i j k : ι), c.Rel i j → c.Rel j k → Eq (CategoryTheory.CategoryStruct. …
    d₂ : (i j : ι) → Quiver.Hom (X₁ i) (X₁ j)
    s₂ : ∀ (i j : ι), Not (c.Rel i j) → Eq (d₂ i j) 0
    h₂ : ∀ (i j k : ι), c.Rel i j → c.Rel j k → Eq (CategoryTheory.CategoryStruct. …
    h_d : ∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp ({ X :=  …
    i j : ι
    ⊢ Eq (d₁ i j) (d₂ i j)
  -/
  by_cases hij : c.Rel i j
    /-
      case pos
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      c : ComplexShape ι
      X₁ : ι → V
      d₁ : (i j : ι) → Quiver.Hom (X₁ i) (X₁ j)
      s₁ : ∀ (i j : ι), Not (c.Rel i j) → Eq (d₁ i j) 0
      h₁ : ∀ (i j k : ι), c.Rel i j → c.Rel j k → Eq (CategoryTheory.CategoryStruct. …
      d₂ : (i j : ι) → Quiver.Hom (X₁ i) (X₁ j)
      s₂ : ∀ (i j : ι), Not (c.Rel i j) → Eq (d₂ i j) 0
      h₂ : ∀ (i j k : ι), c.Rel i j → c.Rel j k → Eq (CategoryTheory.CategoryStruct. …
      h_d : ∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp ({ X :=  …
      i j : ι
      hij : c.Rel i j
      ⊢ Eq (d₁ i j) (d₂ i j)
    -/
  · simpa only [comp_id, id_comp, eqToHom_refl] using h_d i j hij
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      c : ComplexShape ι
      X₁ : ι → V
      d₁ : (i j : ι) → Quiver.Hom (X₁ i) (X₁ j)
      s₁ : ∀ (i j : ι), Not (c.Rel i j) → Eq (d₁ i j) 0
      h₁ : ∀ (i j k : ι), c.Rel i j → c.Rel j k → Eq (CategoryTheory.CategoryStruct. …
      d₂ : (i j : ι) → Quiver.Hom (X₁ i) (X₁ j)
      s₂ : ∀ (i j : ι), Not (c.Rel i j) → Eq (d₂ i j) 0
      h₂ : ∀ (i j k : ι), c.Rel i j → c.Rel j k → Eq (CategoryTheory.CategoryStruct. …
      h_d : ∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp ({ X :=  …
      i j : ι
      hij : Not (c.Rel i j)
      ⊢ Eq (d₁ i j) (d₂ i j)
    -/
  · rw [s₁ i j hij, s₂ i j hij]
    /-
      🎉 no goals
    -/


/-- The obvious isomorphism `K.X p ≅ K.X q` when `p = q`. -/
def XIsoOfEq (K : HomologicalComplex V c) {p q : ι} (h : p = q) : K.X p ≅ K.X q :=
              /-
                ι : Type u_1
                V : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} V
                inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                c : ComplexShape ι
                K : HomologicalComplex V c
                p q : ι
                h : Eq p q
                ⊢ Eq (K.X p) (K.X q)
              -/
  eqToIso (by rw [h])
              /-
                🎉 no goals
              -/


@[simp]
lemma XIsoOfEq_rfl (K : HomologicalComplex V c) (p : ι) :
    K.XIsoOfEq (rfl : p = p) = Iso.refl _ := rfl


@[reassoc (attr := simp)]
lemma XIsoOfEq_hom_comp_XIsoOfEq_hom (K : HomologicalComplex V c) {p₁ p₂ p₃ : ι}
    (h₁₂ : p₁ = p₂) (h₂₃ : p₂ = p₃) :
    (K.XIsoOfEq h₁₂).hom ≫ (K.XIsoOfEq h₂₃).hom = (K.XIsoOfEq (h₁₂.trans h₂₃)).hom := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    K : HomologicalComplex V c
    p₁ p₂ p₃ : ι
    h₁₂ : Eq p₁ p₂
    h₂₃ : Eq p₂ p₃
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.XIsoOfEq h₁₂).hom (K.XIsoOfEq h₂₃) …
  -/
  dsimp [XIsoOfEq]
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    K : HomologicalComplex V c
    p₁ p₂ p₃ : ι
    h₁₂ : Eq p₁ p₂
    h₂₃ : Eq p₂ p₃
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
  -/
  simp only [eqToHom_trans]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma XIsoOfEq_hom_comp_XIsoOfEq_inv (K : HomologicalComplex V c) {p₁ p₂ p₃ : ι}
    (h₁₂ : p₁ = p₂) (h₃₂ : p₃ = p₂) :
    (K.XIsoOfEq h₁₂).hom ≫ (K.XIsoOfEq h₃₂).inv = (K.XIsoOfEq (h₁₂.trans h₃₂.symm)).hom := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    K : HomologicalComplex V c
    p₁ p₂ p₃ : ι
    h₁₂ : Eq p₁ p₂
    h₃₂ : Eq p₃ p₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.XIsoOfEq h₁₂).hom (K.XIsoOfEq h₃₂) …
  -/
  dsimp [XIsoOfEq]
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    K : HomologicalComplex V c
    p₁ p₂ p₃ : ι
    h₁₂ : Eq p₁ p₂
    h₃₂ : Eq p₃ p₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
  -/
  simp only [eqToHom_trans]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma XIsoOfEq_inv_comp_XIsoOfEq_hom (K : HomologicalComplex V c) {p₁ p₂ p₃ : ι}
    (h₂₁ : p₂ = p₁) (h₂₃ : p₂ = p₃) :
    (K.XIsoOfEq h₂₁).inv ≫ (K.XIsoOfEq h₂₃).hom = (K.XIsoOfEq (h₂₁.symm.trans h₂₃)).hom := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    K : HomologicalComplex V c
    p₁ p₂ p₃ : ι
    h₂₁ : Eq p₂ p₁
    h₂₃ : Eq p₂ p₃
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.XIsoOfEq h₂₁).inv (K.XIsoOfEq h₂₃) …
  -/
  dsimp [XIsoOfEq]
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    K : HomologicalComplex V c
    p₁ p₂ p₃ : ι
    h₂₁ : Eq p₂ p₁
    h₂₃ : Eq p₂ p₃
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
  -/
  simp only [eqToHom_trans]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma XIsoOfEq_inv_comp_XIsoOfEq_inv (K : HomologicalComplex V c) {p₁ p₂ p₃ : ι}
    (h₂₁ : p₂ = p₁) (h₃₂ : p₃ = p₂) :
    (K.XIsoOfEq h₂₁).inv ≫ (K.XIsoOfEq h₃₂).inv = (K.XIsoOfEq (h₃₂.trans h₂₁).symm).hom := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    K : HomologicalComplex V c
    p₁ p₂ p₃ : ι
    h₂₁ : Eq p₂ p₁
    h₃₂ : Eq p₃ p₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.XIsoOfEq h₂₁).inv (K.XIsoOfEq h₃₂) …
  -/
  dsimp [XIsoOfEq]
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    K : HomologicalComplex V c
    p₁ p₂ p₃ : ι
    h₂₁ : Eq p₂ p₁
    h₃₂ : Eq p₃ p₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
  -/
  simp only [eqToHom_trans]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma XIsoOfEq_hom_comp_d (K : HomologicalComplex V c) {p₁ p₂ : ι} (h : p₁ = p₂) (p₃ : ι) :
                                                     /-
                                                       ι : Type u_1
                                                       V : Type u
                                                       inst✝¹ : CategoryTheory.Category.{v, u} V
                                                       inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                                       c : ComplexShape ι
                                                       K : HomologicalComplex V c
                                                       p₁ p₂ : ι
                                                       h : Eq p₁ p₂
                                                       p₃ : ι
                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.XIsoOfEq h).hom (K.d p₂ p₃)) (K.d  …
                                                     -/
    (K.XIsoOfEq h).hom ≫ K.d p₂ p₃ = K.d p₁ p₃ := by subst h; simp
                                                              /-
                                                                🎉 no goals
                                                              -/


@[reassoc (attr := simp)]
lemma XIsoOfEq_inv_comp_d (K : HomologicalComplex V c) {p₂ p₁ : ι} (h : p₂ = p₁) (p₃ : ι) :
                                                     /-
                                                       ι : Type u_1
                                                       V : Type u
                                                       inst✝¹ : CategoryTheory.Category.{v, u} V
                                                       inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                                       c : ComplexShape ι
                                                       K : HomologicalComplex V c
                                                       p₂ p₁ : ι
                                                       h : Eq p₂ p₁
                                                       p₃ : ι
                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.XIsoOfEq h).inv (K.d p₂ p₃)) (K.d  …
                                                     -/
    (K.XIsoOfEq h).inv ≫ K.d p₂ p₃ = K.d p₁ p₃ := by subst h; simp
                                                              /-
                                                                🎉 no goals
                                                              -/


@[reassoc (attr := simp)]
lemma d_comp_XIsoOfEq_hom (K : HomologicalComplex V c) {p₂ p₃ : ι} (h : p₂ = p₃) (p₁ : ι) :
                                                     /-
                                                       ι : Type u_1
                                                       V : Type u
                                                       inst✝¹ : CategoryTheory.Category.{v, u} V
                                                       inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                                       c : ComplexShape ι
                                                       K : HomologicalComplex V c
                                                       p₂ p₃ : ι
                                                       h : Eq p₂ p₃
                                                       p₁ : ι
                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d p₁ p₂) (K.XIsoOfEq h).hom) (K.d  …
                                                     -/
    K.d p₁ p₂ ≫ (K.XIsoOfEq h).hom = K.d p₁ p₃ := by subst h; simp
                                                              /-
                                                                🎉 no goals
                                                              -/


@[reassoc (attr := simp)]
lemma d_comp_XIsoOfEq_inv (K : HomologicalComplex V c) {p₂ p₃ : ι} (h : p₃ = p₂) (p₁ : ι) :
                                                     /-
                                                       ι : Type u_1
                                                       V : Type u
                                                       inst✝¹ : CategoryTheory.Category.{v, u} V
                                                       inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                                       c : ComplexShape ι
                                                       K : HomologicalComplex V c
                                                       p₂ p₃ : ι
                                                       h : Eq p₃ p₂
                                                       p₁ : ι
                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d p₁ p₂) (K.XIsoOfEq h).inv) (K.d  …
                                                     -/
    K.d p₁ p₂ ≫ (K.XIsoOfEq h).inv = K.d p₁ p₃ := by subst h; simp
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- An `α`-indexed chain complex is a `HomologicalComplex`
in which `d i j ≠ 0` only if `j + 1 = i`.
-/
abbrev ChainComplex (α : Type*) [AddRightCancelSemigroup α] [One α] : Type _ :=
  HomologicalComplex V (ComplexShape.down α)


/-- An `α`-indexed cochain complex is a `HomologicalComplex`
in which `d i j ≠ 0` only if `i + 1 = j`.
-/
abbrev CochainComplex (α : Type*) [AddRightCancelSemigroup α] [One α] : Type _ :=
  HomologicalComplex V (ComplexShape.up α)


@[simp]
theorem prev (α : Type*) [AddRightCancelSemigroup α] [One α] (i : α) :
    (ComplexShape.down α).prev i = i + 1 :=
  (ComplexShape.down α).prev_eq' rfl


@[simp]
theorem next (α : Type*) [AddGroup α] [One α] (i : α) : (ComplexShape.down α).next i = i - 1 :=
  (ComplexShape.down α).next_eq' <| sub_add_cancel _ _


@[simp]
theorem next_nat_zero : (ComplexShape.down ℕ).next 0 = 0 := by
  classical
    refine dif_neg ?_
    push_neg
    intro
    apply Nat.noConfusion


@[simp]
theorem next_nat_succ (i : ℕ) : (ComplexShape.down ℕ).next (i + 1) = i :=
  (ComplexShape.down ℕ).next_eq' rfl


@[simp]
theorem prev (α : Type*) [AddGroup α] [One α] (i : α) : (ComplexShape.up α).prev i = i - 1 :=
  (ComplexShape.up α).prev_eq' <| sub_add_cancel _ _


@[simp]
theorem next (α : Type*) [AddRightCancelSemigroup α] [One α] (i : α) :
    (ComplexShape.up α).next i = i + 1 :=
  (ComplexShape.up α).next_eq' rfl


@[simp]
theorem prev_nat_zero : (ComplexShape.up ℕ).prev 0 = 0 := by
  classical
    refine dif_neg ?_
    push_neg
    intro
    apply Nat.noConfusion


@[simp]
theorem prev_nat_succ (i : ℕ) : (ComplexShape.up ℕ).prev (i + 1) = i :=
  (ComplexShape.up ℕ).prev_eq' rfl


/-- A morphism of homological complexes consists of maps between the chain groups,
commuting with the differentials.
-/
@[ext]
structure Hom (A B : HomologicalComplex V c) where
  f : ∀ i, A.X i ⟶ B.X i
  comm' : ∀ i j, c.Rel i j → f i ≫ B.d i j = A.d i j ≫ f j := by aesop_cat


@[reassoc (attr := simp)]
theorem Hom.comm {A B : HomologicalComplex V c} (f : A.Hom B) (i j : ι) :
    f.f i ≫ B.d i j = A.d i j ≫ f.f j := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    A B : HomologicalComplex V c
    f : A.Hom B
    i j : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.f i) (B.d i j)) (CategoryTheory.Ca …
  -/
  by_cases hij : c.Rel i j
    /-
      case pos
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      c : ComplexShape ι
      A B : HomologicalComplex V c
      f : A.Hom B
      i j : ι
      hij : c.Rel i j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.f i) (B.d i j)) (CategoryTheory.Ca …
    -/
  · exact f.comm' i j hij
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      c : ComplexShape ι
      A B : HomologicalComplex V c
      f : A.Hom B
      i j : ι
      hij : Not (c.Rel i j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.f i) (B.d i j)) (CategoryTheory.Ca …
    -/
  · rw [A.shape i j hij, B.shape i j hij, comp_zero, zero_comp]
    /-
      🎉 no goals
    -/


instance (A B : HomologicalComplex V c) : Inhabited (Hom A B) :=
  ⟨{ f := fun _ => 0 }⟩


/-- Identity chain map. -/
def id (A : HomologicalComplex V c) : Hom A A where f _ := 𝟙 _


/-- Composition of chain maps. -/
def comp (A B C : HomologicalComplex V c) (φ : Hom A B) (ψ : Hom B C) : Hom A C where
  f i := φ.f i ≫ ψ.f i


attribute [local simp] id comp


instance : Category (HomologicalComplex V c) where
  Hom := Hom
  id := id
  comp := comp _ _ _


@[ext]
lemma hom_ext {C D : HomologicalComplex V c} (f g : C ⟶ D)
    (h : ∀ i, f.f i = g.f i) : f = g := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    f g : Quiver.Hom C D
    h : ∀ (i : ι), Eq (f.f i) (g.f i)
    ⊢ Eq f g
  -/
  apply Hom.ext
  /-
    case f
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    f g : Quiver.Hom C D
    h : ∀ (i : ι), Eq (f.f i) (g.f i)
    ⊢ Eq f.f g.f
  -/
  funext
  /-
    case f.h
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    f g : Quiver.Hom C D
    h : ∀ (i : ι), Eq (f.f i) (g.f i)
    x✝ : ι
    ⊢ Eq (f.f x✝) (g.f x✝)
  -/
  apply h
  /-
    🎉 no goals
  -/


@[simp]
theorem id_f (C : HomologicalComplex V c) (i : ι) : Hom.f (𝟙 C) i = 𝟙 (C.X i) :=
  rfl


@[simp, reassoc]
theorem comp_f {C₁ C₂ C₃ : HomologicalComplex V c} (f : C₁ ⟶ C₂) (g : C₂ ⟶ C₃) (i : ι) :
    (f ≫ g).f i = f.f i ≫ g.f i :=
  rfl


@[simp]
theorem eqToHom_f {C₁ C₂ : HomologicalComplex V c} (h : C₁ = C₂) (n : ι) :
    HomologicalComplex.Hom.f (eqToHom h) n =
      eqToHom (congr_fun (congr_arg HomologicalComplex.X h) n) := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C₁ C₂ : HomologicalComplex V c
    h : Eq C₁ C₂
    n : ι
    ⊢ Eq ((CategoryTheory.eqToHom h).f n) (CategoryTheory.eqToHom ⋯)
  -/
  subst h
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C₁ : HomologicalComplex V c
    n : ι
    ⊢ Eq ((CategoryTheory.eqToHom ⋯).f n) (CategoryTheory.eqToHom ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/

-- We'll use this later to show that `HomologicalComplex V c` is preadditive when `V` is.

theorem hom_f_injective {C₁ C₂ : HomologicalComplex V c} :
                                                      /-
                                                        ι : Type u_1
                                                        V : Type u
                                                        inst✝¹ : CategoryTheory.Category.{v, u} V
                                                        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                                        c : ComplexShape ι
                                                        C₁ C₂ : HomologicalComplex V c
                                                        ⊢ Function.Injective fun f => f.f
                                                      -/
    Function.Injective fun f : Hom C₁ C₂ => f.f := by aesop_cat
                                                      /-
                                                        🎉 no goals
                                                      -/


instance (X Y : HomologicalComplex V c) : Zero (X ⟶ Y) :=
  ⟨{ f := fun _ => 0}⟩


@[simp]
theorem zero_f (C D : HomologicalComplex V c) (i : ι) : (0 : C ⟶ D).f i = 0 :=
  rfl


instance : HasZeroMorphisms (HomologicalComplex V c) where


/-- The zero complex -/
noncomputable def zero [HasZeroObject V] : HomologicalComplex V c where
  X _ := 0
  d _ _ := 0


theorem isZero_zero [HasZeroObject V] : IsZero (zero : HomologicalComplex V c) := by
  /-
    ι : Type u_1
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    inst✝ : CategoryTheory.Limits.HasZeroObject V
    ⊢ CategoryTheory.Limits.IsZero HomologicalComplex.zero
  -/
  refine ⟨fun X => ⟨⟨⟨0⟩, fun f => ?_⟩⟩, fun X => ⟨⟨⟨0⟩, fun f => ?_⟩⟩⟩
  all_goals
    ext
    dsimp only [zero]
    subsingleton


instance [HasZeroObject V] : HasZeroObject (HomologicalComplex V c) :=
  ⟨⟨zero, isZero_zero⟩⟩


noncomputable instance [HasZeroObject V] : Inhabited (HomologicalComplex V c) :=
  ⟨zero⟩


theorem congr_hom {C D : HomologicalComplex V c} {f g : C ⟶ D} (w : f = g) (i : ι) :
    f.f i = g.f i :=
  congr_fun (congr_arg Hom.f w) i


lemma mono_of_mono_f {K L : HomologicalComplex V c} (φ : K ⟶ L)
    (hφ : ∀ i, Mono (φ.f i)) : Mono φ where
  right_cancellation g h eq := by
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      c : ComplexShape ι
      K L : HomologicalComplex V c
      φ : Quiver.Hom K L
      hφ : ∀ (i : ι), CategoryTheory.Mono (φ.f i)
      Z✝ : HomologicalComplex V c
      g h : Quiver.Hom Z✝ K
      eq : Eq (CategoryTheory.CategoryStruct.comp g φ) (CategoryTheory.CategoryStruc …
      ⊢ Eq g h
    -/
    ext i
    /-
      case h
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      c : ComplexShape ι
      K L : HomologicalComplex V c
      φ : Quiver.Hom K L
      hφ : ∀ (i : ι), CategoryTheory.Mono (φ.f i)
      Z✝ : HomologicalComplex V c
      g h : Quiver.Hom Z✝ K
      eq : Eq (CategoryTheory.CategoryStruct.comp g φ) (CategoryTheory.CategoryStruc …
      i : ι
      ⊢ Eq (g.f i) (h.f i)
    -/
    rw [← cancel_mono (φ.f i)]
    /-
      case h
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      c : ComplexShape ι
      K L : HomologicalComplex V c
      φ : Quiver.Hom K L
      hφ : ∀ (i : ι), CategoryTheory.Mono (φ.f i)
      Z✝ : HomologicalComplex V c
      g h : Quiver.Hom Z✝ K
      eq : Eq (CategoryTheory.CategoryStruct.comp g φ) (CategoryTheory.CategoryStruc …
      i : ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (g.f i) (φ.f i)) (CategoryTheory.Cate …
    -/
    exact congr_hom eq i
    /-
      🎉 no goals
    -/


lemma epi_of_epi_f {K L : HomologicalComplex V c} (φ : K ⟶ L)
    (hφ : ∀ i, Epi (φ.f i)) : Epi φ where
  left_cancellation g h eq := by
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      c : ComplexShape ι
      K L : HomologicalComplex V c
      φ : Quiver.Hom K L
      hφ : ∀ (i : ι), CategoryTheory.Epi (φ.f i)
      Z✝ : HomologicalComplex V c
      g h : Quiver.Hom L Z✝
      eq : Eq (CategoryTheory.CategoryStruct.comp φ g) (CategoryTheory.CategoryStruc …
      ⊢ Eq g h
    -/
    ext i
    /-
      case h
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      c : ComplexShape ι
      K L : HomologicalComplex V c
      φ : Quiver.Hom K L
      hφ : ∀ (i : ι), CategoryTheory.Epi (φ.f i)
      Z✝ : HomologicalComplex V c
      g h : Quiver.Hom L Z✝
      eq : Eq (CategoryTheory.CategoryStruct.comp φ g) (CategoryTheory.CategoryStruc …
      i : ι
      ⊢ Eq (g.f i) (h.f i)
    -/
    rw [← cancel_epi (φ.f i)]
    /-
      case h
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      c : ComplexShape ι
      K L : HomologicalComplex V c
      φ : Quiver.Hom K L
      hφ : ∀ (i : ι), CategoryTheory.Epi (φ.f i)
      Z✝ : HomologicalComplex V c
      g h : Quiver.Hom L Z✝
      eq : Eq (CategoryTheory.CategoryStruct.comp φ g) (CategoryTheory.CategoryStruc …
      i : ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (φ.f i) (g.f i)) (CategoryTheory.Cate …
    -/
    exact congr_hom eq i
    /-
      🎉 no goals
    -/


/-- The functor picking out the `i`-th object of a complex. -/
@[simps]
def eval (i : ι) : HomologicalComplex V c ⥤ V where
  obj C := C.X i
  map f := f.f i


instance (i : ι) : (eval V c i).PreservesZeroMorphisms where


/-- The functor forgetting the differential in a complex, obtaining a graded object. -/
@[simps]
def forget : HomologicalComplex V c ⥤ GradedObject ι V where
  obj C := C.X
  map f := f.f


instance : (forget V c).Faithful where
  map_injective h := by
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      c : ComplexShape ι
      C X✝ Y✝ : HomologicalComplex V c
      a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
      h : Eq ((HomologicalComplex.forget V c).map a₁✝) ((HomologicalComplex.forget V …
      ⊢ Eq a₁✝ a₂✝
    -/
    ext i
    /-
      case h
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      c : ComplexShape ι
      C X✝ Y✝ : HomologicalComplex V c
      a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
      h : Eq ((HomologicalComplex.forget V c).map a₁✝) ((HomologicalComplex.forget V …
      i : ι
      ⊢ Eq (a₁✝.f i) (a₂✝.f i)
    -/
    exact congr_fun h i
    /-
      🎉 no goals
    -/


/-- Forgetting the differentials than picking out the `i`-th object is the same as
just picking out the `i`-th object. -/
@[simps!]
def forgetEval (i : ι) : forget V c ⋙ GradedObject.eval i ≅ eval V c i :=
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    i : ι
    ⊢ ∀ {X Y : HomologicalComplex V c} (f : Quiver.Hom X Y), Eq (CategoryTheory.Ca …
  -/
  NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


@[reassoc]
lemma XIsoOfEq_hom_naturality {K L : HomologicalComplex V c} (φ : K ⟶ L) {n n' : ι} (h : n = n') :
                                                                   /-
                                                                     ι : Type u_1
                                                                     V : Type u
                                                                     inst✝¹ : CategoryTheory.Category.{v, u} V
                                                                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                                                     c : ComplexShape ι
                                                                     K L : HomologicalComplex V c
                                                                     φ : Quiver.Hom K L
                                                                     n n' : ι
                                                                     h : Eq n n'
                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (φ.f n) (L.XIsoOfEq h).hom) (Category …
                                                                   -/
    φ.f n ≫ (L.XIsoOfEq h).hom = (K.XIsoOfEq h).hom ≫ φ.f n' := by subst h; simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[reassoc]
lemma XIsoOfEq_inv_naturality {K L : HomologicalComplex V c} (φ : K ⟶ L) {n n' : ι} (h : n = n') :
                                                                   /-
                                                                     ι : Type u_1
                                                                     V : Type u
                                                                     inst✝¹ : CategoryTheory.Category.{v, u} V
                                                                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                                                     c : ComplexShape ι
                                                                     K L : HomologicalComplex V c
                                                                     φ : Quiver.Hom K L
                                                                     n n' : ι
                                                                     h : Eq n n'
                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (φ.f n') (L.XIsoOfEq h).inv) (Categor …
                                                                   -/
    φ.f n' ≫ (L.XIsoOfEq h).inv = (K.XIsoOfEq h).inv ≫ φ.f n := by subst h; simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/

-- Porting note: removed @[simp] as the linter complained

/-- If `C.d i j` and `C.d i j'` are both allowed, then we must have `j = j'`,
and so the differentials only differ by an `eqToHom`.
-/
theorem d_comp_eqToHom {i j j' : ι} (rij : c.Rel i j) (rij' : c.Rel i j') :
    C.d i j' ≫ eqToHom (congr_arg C.X (c.next_eq rij' rij)) = C.d i j := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    i j j' : ι
    rij : c.Rel i j
    rij' : c.Rel i j'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.d i j') (CategoryTheory.eqToHom ⋯) …
  -/
  obtain rfl := c.next_eq rij rij'
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    i j : ι
    rij rij' : c.Rel i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.d i j) (CategoryTheory.eqToHom ⋯)) …
  -/
  simp only [eqToHom_refl, comp_id]
  /-
    🎉 no goals
  -/

-- Porting note: removed @[simp] as the linter complained

/-- If `C.d i j` and `C.d i' j` are both allowed, then we must have `i = i'`,
and so the differentials only differ by an `eqToHom`.
-/
theorem eqToHom_comp_d {i i' j : ι} (rij : c.Rel i j) (rij' : c.Rel i' j) :
    eqToHom (congr_arg C.X (c.prev_eq rij rij')) ≫ C.d i' j = C.d i j := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    i i' j : ι
    rij : c.Rel i j
    rij' : c.Rel i' j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (C.d i' j) …
  -/
  obtain rfl := c.prev_eq rij rij'
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    i j : ι
    rij rij' : c.Rel i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (C.d i j)) …
  -/
  simp only [eqToHom_refl, id_comp]
  /-
    🎉 no goals
  -/


theorem kernel_eq_kernel [HasKernels V] {i j j' : ι} (r : c.Rel i j) (r' : c.Rel i j') :
    kernelSubobject (C.d i j) = kernelSubobject (C.d i j') := by
  /-
    ι : Type u_1
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    inst✝ : CategoryTheory.Limits.HasKernels V
    i j j' : ι
    r : c.Rel i j
    r' : c.Rel i j'
    ⊢ Eq (CategoryTheory.Limits.kernelSubobject (C.d i j)) (CategoryTheory.Limits. …
  -/
  rw [← d_comp_eqToHom C r r']
  /-
    ι : Type u_1
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    inst✝ : CategoryTheory.Limits.HasKernels V
    i j j' : ι
    r : c.Rel i j
    r' : c.Rel i j'
    ⊢ Eq (CategoryTheory.Limits.kernelSubobject (CategoryTheory.CategoryStruct.com …
  -/
  apply kernelSubobject_comp_mono
  /-
    🎉 no goals
  -/


theorem image_eq_image [HasImages V] [HasEqualizers V] {i i' j : ι} (r : c.Rel i j)
    (r' : c.Rel i' j) : imageSubobject (C.d i j) = imageSubobject (C.d i' j) := by
  /-
    ι : Type u_1
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    inst✝¹ : CategoryTheory.Limits.HasImages V
    inst✝ : CategoryTheory.Limits.HasEqualizers V
    i i' j : ι
    r : c.Rel i j
    r' : c.Rel i' j
    ⊢ Eq (CategoryTheory.Limits.imageSubobject (C.d i j)) (CategoryTheory.Limits.i …
  -/
  rw [← eqToHom_comp_d C r r']
  /-
    ι : Type u_1
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    inst✝¹ : CategoryTheory.Limits.HasImages V
    inst✝ : CategoryTheory.Limits.HasEqualizers V
    i i' j : ι
    r : c.Rel i j
    r' : c.Rel i' j
    ⊢ Eq (CategoryTheory.Limits.imageSubobject (CategoryTheory.CategoryStruct.comp …
  -/
  apply imageSubobject_iso_comp
  /-
    🎉 no goals
  -/


/-- Either `C.X i`, if there is some `i` with `c.Rel i j`, or `C.X j`. -/
abbrev xPrev (j : ι) : V :=
  C.X (c.prev j)


/-- If `c.Rel i j`, then `C.xPrev j` is isomorphic to `C.X i`. -/
def xPrevIso {i j : ι} (r : c.Rel i j) : C.xPrev j ≅ C.X i :=
                /-
                  ι : Type u_1
                  V : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} V
                  inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                  c : ComplexShape ι
                  C : HomologicalComplex V c
                  i j : ι
                  r : c.Rel i j
                  ⊢ Eq (C.xPrev j) (C.X i)
                -/
  eqToIso <| by rw [← c.prev_eq' r]
                /-
                  🎉 no goals
                -/


/-- If there is no `i` so `c.Rel i j`, then `C.xPrev j` is isomorphic to `C.X j`. -/
def xPrevIsoSelf {j : ι} (h : ¬c.Rel (c.prev j) j) : C.xPrev j ≅ C.X j :=
  eqToIso <|
    congr_arg C.X
      (by
        /-
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          c : ComplexShape ι
          C : HomologicalComplex V c
          j : ι
          h : Not (c.Rel (c.prev j) j)
          ⊢ Eq (c.prev j) j
        -/
        dsimp [ComplexShape.prev]
        /-
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          c : ComplexShape ι
          C : HomologicalComplex V c
          j : ι
          h : Not (c.Rel (c.prev j) j)
          ⊢ Eq (dite (Exists fun i => c.Rel i j) (fun h => h.choose) fun h => j) j
        -/
        rw [dif_neg]
        /-
          case hnc
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          c : ComplexShape ι
          C : HomologicalComplex V c
          j : ι
          h : Not (c.Rel (c.prev j) j)
          ⊢ Not (Exists fun i => c.Rel i j)
        -/
        push_neg; intro i hi
        /-
          case hnc
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          c : ComplexShape ι
          C : HomologicalComplex V c
          j : ι
          h : Not (c.Rel (c.prev j) j)
          i : ι
          hi : c.Rel i j
          ⊢ False
        -/
        have : c.prev j = i := c.prev_eq' hi
        /-
          case hnc
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          c : ComplexShape ι
          C : HomologicalComplex V c
          j : ι
          h : Not (c.Rel (c.prev j) j)
          i : ι
          hi : c.Rel i j
          this : Eq (c.prev j) i
          ⊢ False
        -/
        rw [this] at h; contradiction)
                        /-
                          🎉 no goals
                        -/


/-- Either `C.X j`, if there is some `j` with `c.rel i j`, or `C.X i`. -/
abbrev xNext (i : ι) : V :=
  C.X (c.next i)


/-- If `c.Rel i j`, then `C.xNext i` is isomorphic to `C.X j`. -/
def xNextIso {i j : ι} (r : c.Rel i j) : C.xNext i ≅ C.X j :=
                /-
                  ι : Type u_1
                  V : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} V
                  inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                  c : ComplexShape ι
                  C : HomologicalComplex V c
                  i j : ι
                  r : c.Rel i j
                  ⊢ Eq (C.xNext i) (C.X j)
                -/
  eqToIso <| by rw [← c.next_eq' r]
                /-
                  🎉 no goals
                -/


/-- If there is no `j` so `c.Rel i j`, then `C.xNext i` is isomorphic to `C.X i`. -/
def xNextIsoSelf {i : ι} (h : ¬c.Rel i (c.next i)) : C.xNext i ≅ C.X i :=
  eqToIso <|
    congr_arg C.X
      (by
        /-
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          c : ComplexShape ι
          C : HomologicalComplex V c
          i : ι
          h : Not (c.Rel i (c.next i))
          ⊢ Eq (c.next i) i
        -/
        dsimp [ComplexShape.next]
        /-
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          c : ComplexShape ι
          C : HomologicalComplex V c
          i : ι
          h : Not (c.Rel i (c.next i))
          ⊢ Eq (dite (Exists fun j => c.Rel i j) (fun h => h.choose) fun h => i) i
        -/
        rw [dif_neg]; rintro ⟨j, hj⟩
        /-
          case hnc.intro
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          c : ComplexShape ι
          C : HomologicalComplex V c
          i : ι
          h : Not (c.Rel i (c.next i))
          j : ι
          hj : c.Rel i j
          ⊢ False
        -/
        have : c.next i = j := c.next_eq' hj
        /-
          case hnc.intro
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
          c : ComplexShape ι
          C : HomologicalComplex V c
          i : ι
          h : Not (c.Rel i (c.next i))
          j : ι
          hj : c.Rel i j
          this : Eq (c.next i) j
          ⊢ False
        -/
        rw [this] at h; contradiction)
                        /-
                          🎉 no goals
                        -/


/-- The differential mapping into `C.X j`, or zero if there isn't one.
-/
abbrev dTo (j : ι) : C.xPrev j ⟶ C.X j :=
  C.d (c.prev j) j


/-- The differential mapping out of `C.X i`, or zero if there isn't one.
-/
abbrev dFrom (i : ι) : C.X i ⟶ C.xNext i :=
  C.d i (c.next i)


theorem dTo_eq {i j : ι} (r : c.Rel i j) : C.dTo j = (C.xPrevIso r).hom ≫ C.d i j := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    i j : ι
    r : c.Rel i j
    ⊢ Eq (C.dTo j) (CategoryTheory.CategoryStruct.comp (C.xPrevIso r).hom (C.d i j))
  -/
  obtain rfl := c.prev_eq' r
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    j : ι
    r : c.Rel (c.prev j) j
    ⊢ Eq (C.dTo j) (CategoryTheory.CategoryStruct.comp (C.xPrevIso r).hom (C.d (c. …
  -/
  exact (Category.id_comp _).symm
  /-
    🎉 no goals
  -/


@[simp]
theorem dTo_eq_zero {j : ι} (h : ¬c.Rel (c.prev j) j) : C.dTo j = 0 :=
  C.shape _ _ h


theorem dFrom_eq {i j : ι} (r : c.Rel i j) : C.dFrom i = C.d i j ≫ (C.xNextIso r).inv := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    i j : ι
    r : c.Rel i j
    ⊢ Eq (C.dFrom i) (CategoryTheory.CategoryStruct.comp (C.d i j) (C.xNextIso r). …
  -/
  obtain rfl := c.next_eq' r
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    i : ι
    r : c.Rel i (c.next i)
    ⊢ Eq (C.dFrom i) (CategoryTheory.CategoryStruct.comp (C.d i (c.next i)) (C.xNe …
  -/
  exact (Category.comp_id _).symm
  /-
    🎉 no goals
  -/


@[simp]
theorem dFrom_eq_zero {i : ι} (h : ¬c.Rel i (c.next i)) : C.dFrom i = 0 :=
  C.shape _ _ h


@[reassoc (attr := simp)]
theorem xPrevIso_comp_dTo {i j : ι} (r : c.Rel i j) : (C.xPrevIso r).inv ≫ C.dTo j = C.d i j := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    i j : ι
    r : c.Rel i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.xPrevIso r).inv (C.dTo j)) (C.d i j)
  -/
  simp [C.dTo_eq r]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem xPrevIsoSelf_comp_dTo {j : ι} (h : ¬c.Rel (c.prev j) j) :
                                               /-
                                                 ι : Type u_1
                                                 V : Type u
                                                 inst✝¹ : CategoryTheory.Category.{v, u} V
                                                 inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                                 c : ComplexShape ι
                                                 C : HomologicalComplex V c
                                                 j : ι
                                                 h : Not (c.Rel (c.prev j) j)
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.xPrevIsoSelf h).inv (C.dTo j)) 0
                                               -/
    (C.xPrevIsoSelf h).inv ≫ C.dTo j = 0 := by simp [h]
                                               /-
                                                 🎉 no goals
                                               -/


@[reassoc (attr := simp)]
theorem dFrom_comp_xNextIso {i j : ι} (r : c.Rel i j) :
    C.dFrom i ≫ (C.xNextIso r).hom = C.d i j := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    i j : ι
    r : c.Rel i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.dFrom i) (C.xNextIso r).hom) (C.d  …
  -/
  simp [C.dFrom_eq r]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem dFrom_comp_xNextIsoSelf {i : ι} (h : ¬c.Rel i (c.next i)) :
                                                 /-
                                                   ι : Type u_1
                                                   V : Type u
                                                   inst✝¹ : CategoryTheory.Category.{v, u} V
                                                   inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                                   c : ComplexShape ι
                                                   C : HomologicalComplex V c
                                                   i : ι
                                                   h : Not (c.Rel i (c.next i))
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.dFrom i) (C.xNextIsoSelf h).hom) 0
                                                 -/
    C.dFrom i ≫ (C.xNextIsoSelf h).hom = 0 := by simp [h]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp 1100]
theorem dTo_comp_dFrom (j : ι) : C.dTo j ≫ C.dFrom j = 0 :=
  C.d_comp_d _ _ _


theorem kernel_from_eq_kernel [HasKernels V] {i j : ι} (r : c.Rel i j) :
    kernelSubobject (C.dFrom i) = kernelSubobject (C.d i j) := by
  /-
    ι : Type u_1
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    inst✝ : CategoryTheory.Limits.HasKernels V
    i j : ι
    r : c.Rel i j
    ⊢ Eq (CategoryTheory.Limits.kernelSubobject (C.dFrom i)) (CategoryTheory.Limit …
  -/
  rw [C.dFrom_eq r]
  /-
    ι : Type u_1
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    inst✝ : CategoryTheory.Limits.HasKernels V
    i j : ι
    r : c.Rel i j
    ⊢ Eq (CategoryTheory.Limits.kernelSubobject (CategoryTheory.CategoryStruct.com …
  -/
  apply kernelSubobject_comp_mono
  /-
    🎉 no goals
  -/


theorem image_to_eq_image [HasImages V] [HasEqualizers V] {i j : ι} (r : c.Rel i j) :
    imageSubobject (C.dTo j) = imageSubobject (C.d i j) := by
  /-
    ι : Type u_1
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    inst✝¹ : CategoryTheory.Limits.HasImages V
    inst✝ : CategoryTheory.Limits.HasEqualizers V
    i j : ι
    r : c.Rel i j
    ⊢ Eq (CategoryTheory.Limits.imageSubobject (C.dTo j)) (CategoryTheory.Limits.i …
  -/
  rw [C.dTo_eq r]
  /-
    ι : Type u_1
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C : HomologicalComplex V c
    inst✝¹ : CategoryTheory.Limits.HasImages V
    inst✝ : CategoryTheory.Limits.HasEqualizers V
    i j : ι
    r : c.Rel i j
    ⊢ Eq (CategoryTheory.Limits.imageSubobject (CategoryTheory.CategoryStruct.comp …
  -/
  apply imageSubobject_iso_comp
  /-
    🎉 no goals
  -/


/-- The `i`-th component of an isomorphism of chain complexes. -/
@[simps!]
def isoApp (f : C₁ ≅ C₂) (i : ι) : C₁.X i ≅ C₂.X i :=
  (eval V c i).mapIso f


/-- Construct an isomorphism of chain complexes from isomorphism of the objects
which commute with the differentials. -/
@[simps]
def isoOfComponents (f : ∀ i, C₁.X i ≅ C₂.X i)
    (hf : ∀ i j, c.Rel i j → (f i).hom ≫ C₂.d i j = C₁.d i j ≫ (f j).hom := by aesop_cat) :
    C₁ ≅ C₂ where
  hom :=
    { f := fun i => (f i).hom
      comm' := hf }
  inv :=
    { f := fun i => (f i).inv
      comm' := fun i j hij =>
        calc
                                                                                      /-
                                                                                        ι : Type u_1
                                                                                        V : Type u
                                                                                        inst✝¹ : CategoryTheory.Category.{v, u} V
                                                                                        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                                                                        c : ComplexShape ι
                                                                                        C C₁ C₂ C₃ : HomologicalComplex V c
                                                                                        f : (i : ι) → CategoryTheory.Iso (C₁.X i) (C₂.X i)
                                                                                        hf : autoParam (∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.com …
                                                                                        i j : ι
                                                                                        hij : c.Rel i j
                                                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (f i).inv (C₁.d i j)) (CategoryTheory …
                                                                                      -/
          (f i).inv ≫ C₁.d i j = (f i).inv ≫ (C₁.d i j ≫ (f j).hom) ≫ (f j).inv := by simp
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
                                                                   /-
                                                                     ι : Type u_1
                                                                     V : Type u
                                                                     inst✝¹ : CategoryTheory.Category.{v, u} V
                                                                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                                                     c : ComplexShape ι
                                                                     C C₁ C₂ C₃ : HomologicalComplex V c
                                                                     f : (i : ι) → CategoryTheory.Iso (C₁.X i) (C₂.X i)
                                                                     hf : autoParam (∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.com …
                                                                     i j : ι
                                                                     hij : c.Rel i j
                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (f i).inv (CategoryTheory.CategoryStr …
                                                                   -/
          _ = (f i).inv ≫ ((f i).hom ≫ C₂.d i j) ≫ (f j).inv := by rw [hf i j hij]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                         /-
                                           ι : Type u_1
                                           V : Type u
                                           inst✝¹ : CategoryTheory.Category.{v, u} V
                                           inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                           c : ComplexShape ι
                                           C C₁ C₂ C₃ : HomologicalComplex V c
                                           f : (i : ι) → CategoryTheory.Iso (C₁.X i) (C₂.X i)
                                           hf : autoParam (∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.com …
                                           i j : ι
                                           hij : c.Rel i j
                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (f i).inv (CategoryTheory.CategoryStr …
                                         -/
          _ = C₂.d i j ≫ (f j).inv := by simp }
                                         /-
                                           🎉 no goals
                                         -/
  hom_inv_id := by
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      c : ComplexShape ι
      C C₁ C₂ C₃ : HomologicalComplex V c
      f : (i : ι) → CategoryTheory.Iso (C₁.X i) (C₂.X i)
      hf : autoParam (∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.com …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { f := fun i => (f i).hom, comm' := h …
    -/
    ext i
    /-
      case h
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      c : ComplexShape ι
      C C₁ C₂ C₃ : HomologicalComplex V c
      f : (i : ι) → CategoryTheory.Iso (C₁.X i) (C₂.X i)
      hf : autoParam (∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.com …
      i : ι
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { f := fun i => (f i).hom, comm' :=  …
    -/
    exact (f i).hom_inv_id
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      c : ComplexShape ι
      C C₁ C₂ C₃ : HomologicalComplex V c
      f : (i : ι) → CategoryTheory.Iso (C₁.X i) (C₂.X i)
      hf : autoParam (∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.com …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { f := fun i => (f i).inv, comm' := ⋯ …
    -/
    ext i
    /-
      case h
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      c : ComplexShape ι
      C C₁ C₂ C₃ : HomologicalComplex V c
      f : (i : ι) → CategoryTheory.Iso (C₁.X i) (C₂.X i)
      hf : autoParam (∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.com …
      i : ι
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { f := fun i => (f i).inv, comm' :=  …
    -/
    exact (f i).inv_hom_id
    /-
      🎉 no goals
    -/


@[simp]
theorem isoOfComponents_app (f : ∀ i, C₁.X i ≅ C₂.X i)
    (hf : ∀ i j, c.Rel i j → (f i).hom ≫ C₂.d i j = C₁.d i j ≫ (f j).hom) (i : ι) :
    isoApp (isoOfComponents f hf) i = f i := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C₁ C₂ : HomologicalComplex V c
    f : (i : ι) → CategoryTheory.Iso (C₁.X i) (C₂.X i)
    hf : ∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp (f i).hom …
    i : ι
    ⊢ Eq (HomologicalComplex.Hom.isoApp (HomologicalComplex.Hom.isoOfComponents f  …
  -/
  ext
  /-
    case w
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C₁ C₂ : HomologicalComplex V c
    f : (i : ι) → CategoryTheory.Iso (C₁.X i) (C₂.X i)
    hf : ∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp (f i).hom …
    i : ι
    ⊢ Eq (HomologicalComplex.Hom.isoApp (HomologicalComplex.Hom.isoOfComponents f  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem isIso_of_components (f : C₁ ⟶ C₂) [∀ n : ι, IsIso (f.f n)] : IsIso f :=
   /-
     ι : Type u_1
     V : Type u
     inst✝² : CategoryTheory.Category.{v, u} V
     inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
     c : ComplexShape ι
     C₁ C₂ : HomologicalComplex V c
     f : Quiver.Hom C₁ C₂
     inst✝ : ∀ (n : ι), CategoryTheory.IsIso (f.f n)
     ⊢ ∀ (i j : ι), c.Rel i j → Eq (CategoryTheory.CategoryStruct.comp ((fun n => C …
   -/
  (HomologicalComplex.Hom.isoOfComponents fun n => asIso (f.f n)).isIso_hom
   /-
     🎉 no goals
   -/


/-- `f.prev j` is `f.f i` if there is some `r i j`, and `f.f j` otherwise. -/
abbrev prev (f : Hom C₁ C₂) (j : ι) : C₁.xPrev j ⟶ C₂.xPrev j :=
  f.f _


theorem prev_eq (f : Hom C₁ C₂) {i j : ι} (w : c.Rel i j) :
    f.prev j = (C₁.xPrevIso w).hom ≫ f.f i ≫ (C₂.xPrevIso w).inv := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C₁ C₂ : HomologicalComplex V c
    f : C₁.Hom C₂
    i j : ι
    w : c.Rel i j
    ⊢ Eq (f.prev j) (CategoryTheory.CategoryStruct.comp (C₁.xPrevIso w).hom (Categ …
  -/
  obtain rfl := c.prev_eq' w
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C₁ C₂ : HomologicalComplex V c
    f : C₁.Hom C₂
    j : ι
    w : c.Rel (c.prev j) j
    ⊢ Eq (f.prev j) (CategoryTheory.CategoryStruct.comp (C₁.xPrevIso w).hom (Categ …
  -/
  simp only [xPrevIso, eqToIso_refl, Iso.refl_hom, Iso.refl_inv, comp_id, id_comp]
  /-
    🎉 no goals
  -/


/-- `f.next i` is `f.f j` if there is some `r i j`, and `f.f j` otherwise. -/
abbrev next (f : Hom C₁ C₂) (i : ι) : C₁.xNext i ⟶ C₂.xNext i :=
  f.f _


theorem next_eq (f : Hom C₁ C₂) {i j : ι} (w : c.Rel i j) :
    f.next i = (C₁.xNextIso w).hom ≫ f.f j ≫ (C₂.xNextIso w).inv := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C₁ C₂ : HomologicalComplex V c
    f : C₁.Hom C₂
    i j : ι
    w : c.Rel i j
    ⊢ Eq (f.next i) (CategoryTheory.CategoryStruct.comp (C₁.xNextIso w).hom (Categ …
  -/
  obtain rfl := c.next_eq' w
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    c : ComplexShape ι
    C₁ C₂ : HomologicalComplex V c
    f : C₁.Hom C₂
    i : ι
    w : c.Rel i (c.next i)
    ⊢ Eq (f.next i) (CategoryTheory.CategoryStruct.comp (C₁.xNextIso w).hom (Categ …
  -/
  simp only [xNextIso, eqToIso_refl, Iso.refl_hom, Iso.refl_inv, comp_id, id_comp]
  /-
    🎉 no goals
  -/


@[reassoc, elementwise]
theorem comm_from (f : Hom C₁ C₂) (i : ι) : f.f i ≫ C₂.dFrom i = C₁.dFrom i ≫ f.next i :=
  f.comm _ _


@[reassoc, elementwise]
theorem comm_to (f : Hom C₁ C₂) (j : ι) : f.prev j ≫ C₂.dTo j = C₁.dTo j ≫ f.f j :=
  f.comm _ _


/-- A morphism of chain complexes
induces a morphism of arrows of the differentials out of each object.
-/
def sqFrom (f : Hom C₁ C₂) (i : ι) : Arrow.mk (C₁.dFrom i) ⟶ Arrow.mk (C₂.dFrom i) :=
  Arrow.homMk (f.comm_from i)


@[simp]
theorem sqFrom_left (f : Hom C₁ C₂) (i : ι) : (f.sqFrom i).left = f.f i :=
  rfl


@[simp]
theorem sqFrom_right (f : Hom C₁ C₂) (i : ι) : (f.sqFrom i).right = f.next i :=
  rfl


@[simp]
theorem sqFrom_id (C₁ : HomologicalComplex V c) (i : ι) : sqFrom (𝟙 C₁) i = 𝟙 _ :=
  rfl


@[simp]
theorem sqFrom_comp (f : C₁ ⟶ C₂) (g : C₂ ⟶ C₃) (i : ι) :
    sqFrom (f ≫ g) i = sqFrom f i ≫ sqFrom g i :=
  rfl


/-- A morphism of chain complexes
induces a morphism of arrows of the differentials into each object.
-/
def sqTo (f : Hom C₁ C₂) (j : ι) : Arrow.mk (C₁.dTo j) ⟶ Arrow.mk (C₂.dTo j) :=
  Arrow.homMk (f.comm_to j)


@[simp]
theorem sqTo_left (f : Hom C₁ C₂) (j : ι) : (f.sqTo j).left = f.prev j :=
  rfl


@[simp]
theorem sqTo_right (f : Hom C₁ C₂) (j : ι) : (f.sqTo j).right = f.f j :=
  rfl


/-- Construct an `α`-indexed chain complex from a dependently-typed differential.
-/
def of (X : α → V) (d : ∀ n, X (n + 1) ⟶ X n) (sq : ∀ n, d (n + 1) ≫ d n = 0) : ChainComplex V α :=
  { X := X
                                                      /-
                                                        ι : Type u_1
                                                        V : Type u
                                                        inst✝⁴ : CategoryTheory.Category.{v, u} V
                                                        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
                                                        α : Type u_2
                                                        inst✝² : AddRightCancelSemigroup α
                                                        inst✝¹ : One α
                                                        inst✝ : DecidableEq α
                                                        X : α → V
                                                        d : (n : α) → Quiver.Hom (X (HAdd.hAdd n 1)) (X n)
                                                        sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d (HAdd.hAdd n 1)) (d  …
                                                        i j : α
                                                        h : Eq i (HAdd.hAdd j 1)
                                                        ⊢ Eq (X i) (X (HAdd.hAdd j 1))
                                                      -/
    d := fun i j => if h : i = j + 1 then eqToHom (by rw [h]) ≫ d j else 0
                                                      /-
                                                        🎉 no goals
                                                      -/
    shape := fun i j w => by
      /-
        ι : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
        α : Type u_2
        inst✝² : AddRightCancelSemigroup α
        inst✝¹ : One α
        inst✝ : DecidableEq α
        X : α → V
        d : (n : α) → Quiver.Hom (X (HAdd.hAdd n 1)) (X n)
        sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d (HAdd.hAdd n 1)) (d  …
        i j : α
        w : Not ((ComplexShape.down α).Rel i j)
        ⊢ Eq ((fun i j => dite (Eq i (HAdd.hAdd j 1)) (fun h => CategoryTheory.Categor …
      -/
      dsimp
      /-
        ι : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
        α : Type u_2
        inst✝² : AddRightCancelSemigroup α
        inst✝¹ : One α
        inst✝ : DecidableEq α
        X : α → V
        d : (n : α) → Quiver.Hom (X (HAdd.hAdd n 1)) (X n)
        sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d (HAdd.hAdd n 1)) (d  …
        i j : α
        w : Not ((ComplexShape.down α).Rel i j)
        ⊢ Eq (dite (Eq i (HAdd.hAdd j 1)) (fun h => CategoryTheory.CategoryStruct.comp …
      -/
      rw [dif_neg (Ne.symm w)]
      /-
        🎉 no goals
      -/
    d_comp_d' := fun i j k hij hjk => by
      /-
        ι : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
        α : Type u_2
        inst✝² : AddRightCancelSemigroup α
        inst✝¹ : One α
        inst✝ : DecidableEq α
        X : α → V
        d : (n : α) → Quiver.Hom (X (HAdd.hAdd n 1)) (X n)
        sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d (HAdd.hAdd n 1)) (d  …
        i j k : α
        hij : (ComplexShape.down α).Rel i j
        hjk : (ComplexShape.down α).Rel j k
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i j => dite (Eq i (HAdd.hAdd j  …
      -/
      dsimp at hij hjk
      /-
        ι : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
        α : Type u_2
        inst✝² : AddRightCancelSemigroup α
        inst✝¹ : One α
        inst✝ : DecidableEq α
        X : α → V
        d : (n : α) → Quiver.Hom (X (HAdd.hAdd n 1)) (X n)
        sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d (HAdd.hAdd n 1)) (d  …
        i j k : α
        hij : Eq (HAdd.hAdd j 1) i
        hjk : Eq (HAdd.hAdd k 1) j
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i j => dite (Eq i (HAdd.hAdd j  …
      -/
      substs hij hjk
      /-
        ι : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
        α : Type u_2
        inst✝² : AddRightCancelSemigroup α
        inst✝¹ : One α
        inst✝ : DecidableEq α
        X : α → V
        d : (n : α) → Quiver.Hom (X (HAdd.hAdd n 1)) (X n)
        sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d (HAdd.hAdd n 1)) (d  …
        k : α
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i j => dite (Eq i (HAdd.hAdd j  …
      -/
      simp only [eqToHom_refl, id_comp, dite_eq_ite, ite_true, sq] }
      /-
        🎉 no goals
      -/


@[simp]
theorem of_x (n : α) : (of X d sq).X n = X n :=
  rfl


@[simp]
theorem of_d (j : α) : (of X d sq).d (j + 1) j = d j := by
  /-
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
    α : Type u_2
    inst✝² : AddRightCancelSemigroup α
    inst✝¹ : One α
    inst✝ : DecidableEq α
    X : α → V
    d : (n : α) → Quiver.Hom (X (HAdd.hAdd n 1)) (X n)
    sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d (HAdd.hAdd n 1)) (d  …
    j : α
    ⊢ Eq ((ChainComplex.of X d sq).d (HAdd.hAdd j 1) j) (d j)
  -/
  dsimp [of]
  /-
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
    α : Type u_2
    inst✝² : AddRightCancelSemigroup α
    inst✝¹ : One α
    inst✝ : DecidableEq α
    X : α → V
    d : (n : α) → Quiver.Hom (X (HAdd.hAdd n 1)) (X n)
    sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d (HAdd.hAdd n 1)) (d  …
    j : α
    ⊢ Eq (ite (Eq (HAdd.hAdd j 1) (HAdd.hAdd j 1)) (CategoryTheory.CategoryStruct. …
  -/
  rw [if_pos rfl, Category.id_comp]
  /-
    🎉 no goals
  -/


theorem of_d_ne {i j : α} (h : i ≠ j + 1) : (of X d sq).d i j = 0 := by
  /-
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
    α : Type u_2
    inst✝² : AddRightCancelSemigroup α
    inst✝¹ : One α
    inst✝ : DecidableEq α
    X : α → V
    d : (n : α) → Quiver.Hom (X (HAdd.hAdd n 1)) (X n)
    sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d (HAdd.hAdd n 1)) (d  …
    i j : α
    h : Ne i (HAdd.hAdd j 1)
    ⊢ Eq ((ChainComplex.of X d sq).d i j) 0
  -/
  dsimp [of]
  /-
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
    α : Type u_2
    inst✝² : AddRightCancelSemigroup α
    inst✝¹ : One α
    inst✝ : DecidableEq α
    X : α → V
    d : (n : α) → Quiver.Hom (X (HAdd.hAdd n 1)) (X n)
    sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d (HAdd.hAdd n 1)) (d  …
    i j : α
    h : Ne i (HAdd.hAdd j 1)
    ⊢ Eq (dite (Eq i (HAdd.hAdd j 1)) (fun h => CategoryTheory.CategoryStruct.comp …
  -/
  rw [dif_neg h]
  /-
    🎉 no goals
  -/


/-- A constructor for chain maps between `α`-indexed chain complexes built using `ChainComplex.of`,
from a dependently typed collection of morphisms.
-/
@[simps]
def ofHom (f : ∀ i : α, X i ⟶ Y i) (comm : ∀ i : α, f (i + 1) ≫ d_Y i = d_X i ≫ f i) :
    of X d_X sq_X ⟶ of Y d_Y sq_Y :=
  { f
    comm' := fun n m => by
      /-
        ι : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
        α : Type u_2
        inst✝² : AddRightCancelSemigroup α
        inst✝¹ : One α
        inst✝ : DecidableEq α
        X : α → V
        d_X : (n : α) → Quiver.Hom (X (HAdd.hAdd n 1)) (X n)
        sq_X : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_X (HAdd.hAdd n 1)) …
        Y : α → V
        d_Y : (n : α) → Quiver.Hom (Y (HAdd.hAdd n 1)) (Y n)
        sq_Y : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_Y (HAdd.hAdd n 1)) …
        f : (i : α) → Quiver.Hom (X i) (Y i)
        comm : ∀ (i : α), Eq (CategoryTheory.CategoryStruct.comp (f (HAdd.hAdd i 1)) ( …
        n m : α
        ⊢ (ComplexShape.down α).Rel n m → Eq (CategoryTheory.CategoryStruct.comp (f n) …
      -/
      by_cases h : n = m + 1
        /-
          case pos
          ι : Type u_1
          V : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} V
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
          α : Type u_2
          inst✝² : AddRightCancelSemigroup α
          inst✝¹ : One α
          inst✝ : DecidableEq α
          X : α → V
          d_X : (n : α) → Quiver.Hom (X (HAdd.hAdd n 1)) (X n)
          sq_X : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_X (HAdd.hAdd n 1)) …
          Y : α → V
          d_Y : (n : α) → Quiver.Hom (Y (HAdd.hAdd n 1)) (Y n)
          sq_Y : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_Y (HAdd.hAdd n 1)) …
          f : (i : α) → Quiver.Hom (X i) (Y i)
          comm : ∀ (i : α), Eq (CategoryTheory.CategoryStruct.comp (f (HAdd.hAdd i 1)) ( …
          n m : α
          h : Eq n (HAdd.hAdd m 1)
          ⊢ (ComplexShape.down α).Rel n m → Eq (CategoryTheory.CategoryStruct.comp (f n) …
        -/
      · subst h
        /-
          case pos
          ι : Type u_1
          V : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} V
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
          α : Type u_2
          inst✝² : AddRightCancelSemigroup α
          inst✝¹ : One α
          inst✝ : DecidableEq α
          X : α → V
          d_X : (n : α) → Quiver.Hom (X (HAdd.hAdd n 1)) (X n)
          sq_X : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_X (HAdd.hAdd n 1)) …
          Y : α → V
          d_Y : (n : α) → Quiver.Hom (Y (HAdd.hAdd n 1)) (Y n)
          sq_Y : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_Y (HAdd.hAdd n 1)) …
          f : (i : α) → Quiver.Hom (X i) (Y i)
          comm : ∀ (i : α), Eq (CategoryTheory.CategoryStruct.comp (f (HAdd.hAdd i 1)) ( …
          m : α
          ⊢ (ComplexShape.down α).Rel (HAdd.hAdd m 1) m → Eq (CategoryTheory.CategoryStr …
        -/
        simpa using comm m
        /-
          🎉 no goals
        -/
        /-
          case neg
          ι : Type u_1
          V : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} V
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
          α : Type u_2
          inst✝² : AddRightCancelSemigroup α
          inst✝¹ : One α
          inst✝ : DecidableEq α
          X : α → V
          d_X : (n : α) → Quiver.Hom (X (HAdd.hAdd n 1)) (X n)
          sq_X : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_X (HAdd.hAdd n 1)) …
          Y : α → V
          d_Y : (n : α) → Quiver.Hom (Y (HAdd.hAdd n 1)) (Y n)
          sq_Y : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_Y (HAdd.hAdd n 1)) …
          f : (i : α) → Quiver.Hom (X i) (Y i)
          comm : ∀ (i : α), Eq (CategoryTheory.CategoryStruct.comp (f (HAdd.hAdd i 1)) ( …
          n m : α
          h : Not (Eq n (HAdd.hAdd m 1))
          ⊢ (ComplexShape.down α).Rel n m → Eq (CategoryTheory.CategoryStruct.comp (f n) …
        -/
      · rw [of_d_ne X _ _ h, of_d_ne Y _ _ h]
        /-
          case neg
          ι : Type u_1
          V : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} V
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
          α : Type u_2
          inst✝² : AddRightCancelSemigroup α
          inst✝¹ : One α
          inst✝ : DecidableEq α
          X : α → V
          d_X : (n : α) → Quiver.Hom (X (HAdd.hAdd n 1)) (X n)
          sq_X : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_X (HAdd.hAdd n 1)) …
          Y : α → V
          d_Y : (n : α) → Quiver.Hom (Y (HAdd.hAdd n 1)) (Y n)
          sq_Y : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_Y (HAdd.hAdd n 1)) …
          f : (i : α) → Quiver.Hom (X i) (Y i)
          comm : ∀ (i : α), Eq (CategoryTheory.CategoryStruct.comp (f (HAdd.hAdd i 1)) ( …
          n m : α
          h : Not (Eq n (HAdd.hAdd m 1))
          ⊢ (ComplexShape.down α).Rel n m → Eq (CategoryTheory.CategoryStruct.comp (f n) …
        -/
        simp }
        /-
          🎉 no goals
        -/


/-- Auxiliary definition for `mk`. -/
def mkAux : ℕ → ShortComplex V
  | 0 => ShortComplex.mk _ _ s
  | n + 1 => ShortComplex.mk _ _ (succ (mkAux n)).2.2


/-- An inductive constructor for `ℕ`-indexed chain complexes.

You provide explicitly the first two differentials,
then a function which takes two differentials and the fact they compose to zero,
and returns the next object, its differential, and the fact it composes appropriately to zero.

See also `mk'`, which only sees the previous differential in the inductive step.
-/
def mk : ChainComplex V ℕ :=
  of (fun n => (mkAux X₀ X₁ X₂ d₀ d₁ s succ n).X₃) (fun n => (mkAux X₀ X₁ X₂ d₀ d₁ s succ n).g)
    fun n => (mkAux X₀ X₁ X₂ d₀ d₁ s succ n).zero


@[simp]
theorem mk_X_0 : (mk X₀ X₁ X₂ d₀ d₁ s succ).X 0 = X₀ :=
  rfl


@[simp]
theorem mk_X_1 : (mk X₀ X₁ X₂ d₀ d₁ s succ).X 1 = X₁ :=
  rfl


@[simp]
theorem mk_X_2 : (mk X₀ X₁ X₂ d₀ d₁ s succ).X 2 = X₂ :=
  rfl


@[simp]
theorem mk_d_1_0 : (mk X₀ X₁ X₂ d₀ d₁ s succ).d 1 0 = d₀ := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    X₀ X₁ X₂ : V
    d₀ : Quiver.Hom X₁ X₀
    d₁ : Quiver.Hom X₂ X₁
    s : Eq (CategoryTheory.CategoryStruct.comp d₁ d₀) 0
    succ : (S : CategoryTheory.ShortComplex V) → PSigma fun X₃ => PSigma fun d₂ => …
    ⊢ Eq ((ChainComplex.mk X₀ X₁ X₂ d₀ d₁ s succ).d 1 0) d₀
  -/
  change ite (1 = 0 + 1) (𝟙 X₁ ≫ d₀) 0 = d₀
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    X₀ X₁ X₂ : V
    d₀ : Quiver.Hom X₁ X₀
    d₁ : Quiver.Hom X₂ X₁
    s : Eq (CategoryTheory.CategoryStruct.comp d₁ d₀) 0
    succ : (S : CategoryTheory.ShortComplex V) → PSigma fun X₃ => PSigma fun d₂ => …
    ⊢ Eq (ite (Eq 1 (HAdd.hAdd 0 1)) (CategoryTheory.CategoryStruct.comp (Category …
  -/
  rw [if_pos rfl, Category.id_comp]
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_d_2_1 : (mk X₀ X₁ X₂ d₀ d₁ s succ).d 2 1 = d₁ := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    X₀ X₁ X₂ : V
    d₀ : Quiver.Hom X₁ X₀
    d₁ : Quiver.Hom X₂ X₁
    s : Eq (CategoryTheory.CategoryStruct.comp d₁ d₀) 0
    succ : (S : CategoryTheory.ShortComplex V) → PSigma fun X₃ => PSigma fun d₂ => …
    ⊢ Eq ((ChainComplex.mk X₀ X₁ X₂ d₀ d₁ s succ).d 2 1) d₁
  -/
  change ite (2 = 1 + 1) (𝟙 X₂ ≫ d₁) 0 = d₁
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    X₀ X₁ X₂ : V
    d₀ : Quiver.Hom X₁ X₀
    d₁ : Quiver.Hom X₂ X₁
    s : Eq (CategoryTheory.CategoryStruct.comp d₁ d₀) 0
    succ : (S : CategoryTheory.ShortComplex V) → PSigma fun X₃ => PSigma fun d₂ => …
    ⊢ Eq (ite (Eq 2 (HAdd.hAdd 1 1)) (CategoryTheory.CategoryStruct.comp (Category …
  -/
  rw [if_pos rfl, Category.id_comp]
  /-
    🎉 no goals
  -/

-- TODO simp lemmas for the inductive steps? It's not entirely clear that they are needed.

/-- A simpler inductive constructor for `ℕ`-indexed chain complexes.

You provide explicitly the first differential,
then a function which takes a differential,
and returns the next object, its differential, and the fact it composes appropriately to zero.
-/
def mk' (X₀ X₁ : V) (d : X₁ ⟶ X₀)
    (succ' : ∀ {X₀ X₁ : V} (f : X₁ ⟶ X₀), Σ' (X₂ : V) (d : X₂ ⟶ X₁), d ≫ f = 0) :
    ChainComplex V ℕ :=
  mk _ _ _ _ _ (succ' d).2.2 (fun S => succ' S.f)


@[simp]
theorem mk'_X_0 : (mk' X₀ X₁ d₀ succ').X 0 = X₀ :=
  rfl


@[simp]
theorem mk'_X_1 : (mk' X₀ X₁ d₀ succ').X 1 = X₁ :=
  rfl



@[simp]
theorem mk'_d_1_0 : (mk' X₀ X₁ d₀ succ').d 1 0 = d₀ := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    X₀ X₁ : V
    d₀ : Quiver.Hom X₁ X₀
    succ' : {X₀ X₁ : V} → (f : Quiver.Hom X₁ X₀) → PSigma fun X₂ => PSigma fun d = …
    ⊢ Eq ((ChainComplex.mk' X₀ X₁ d₀ fun {X₀ X₁} => succ').d 1 0) d₀
  -/
  change ite (1 = 0 + 1) (𝟙 X₁ ≫ d₀) 0 = d₀
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    X₀ X₁ : V
    d₀ : Quiver.Hom X₁ X₀
    succ' : {X₀ X₁ : V} → (f : Quiver.Hom X₁ X₀) → PSigma fun X₂ => PSigma fun d = …
    ⊢ Eq (ite (Eq 1 (HAdd.hAdd 0 1)) (CategoryTheory.CategoryStruct.comp (Category …
  -/
  rw [if_pos rfl, Category.id_comp]
  /-
    🎉 no goals
  -/

/- Porting note:
Downstream constructions using `mk'` (e.g. in `CategoryTheory.Abelian.Projective`)
have very slow proofs, because of bad simp lemmas.
It would be better to write good lemmas here if possible, such as

```
theorem mk'_X_succ (j : ℕ) :
    (mk' X₀ X₁ d₀ succ').X (j + 2) = (succ' ⟨_, _, (mk' X₀ X₁ d₀ succ').d (j + 1) j⟩).1 := by
  sorry

theorem mk'_d_succ {i j : ℕ} :
    (mk' X₀ X₁ d₀ succ').d (j + 2) (j + 1) =
      eqToHom (mk'_X_succ X₀ X₁ d₀ succ' j) ≫
      (succ' ⟨_, _, (mk' X₀ X₁ d₀ succ').d (j + 1) j⟩).2.1 :=
  sorry
```

These are already tricky, and it may be better to write analogous lemmas for `mk` first.
-/


/-- An auxiliary construction for `mkHom`.

Here we build by induction a family of commutative squares,
but don't require at the type level that these successive commutative squares actually agree.
They do in fact agree, and we then capture that at the type level (i.e. by constructing a chain map)
in `mkHom`.
-/
def mkHomAux :
    ∀ n,
      Σ' (f : P.X n ⟶ Q.X n) (f' : P.X (n + 1) ⟶ Q.X (n + 1)),
        f' ≫ Q.d (n + 1) n = P.d (n + 1) n ≫ f
  | 0 => ⟨zero, one, one_zero_comm⟩
  | n + 1 => ⟨(mkHomAux n).2.1, (succ n (mkHomAux n)).1, (succ n (mkHomAux n)).2⟩


/-- A constructor for chain maps between `ℕ`-indexed chain complexes,
working by induction on commutative squares.

You need to provide the components of the chain map in degrees 0 and 1,
show that these form a commutative square,
and then give a construction of each component,
and the fact that it forms a commutative square with the previous component,
using as an inductive hypothesis the data (and commutativity) of the previous two components.
-/
def mkHom : P ⟶ Q where
  f n := (mkHomAux P Q zero one one_zero_comm succ n).1
  comm' n m := by
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      P Q : ChainComplex V Nat
      zero : Quiver.Hom (P.X 0) (Q.X 0)
      one : Quiver.Hom (P.X 1) (Q.X 1)
      one_zero_comm : Eq (CategoryTheory.CategoryStruct.comp one (Q.d 1 0)) (Categor …
      succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (CategoryTheory.Ca …
      n m : Nat
      ⊢ (ComplexShape.down Nat).Rel n m → Eq (CategoryTheory.CategoryStruct.comp ((f …
    -/
    rintro (rfl : m + 1 = n)
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      P Q : ChainComplex V Nat
      zero : Quiver.Hom (P.X 0) (Q.X 0)
      one : Quiver.Hom (P.X 1) (Q.X 1)
      one_zero_comm : Eq (CategoryTheory.CategoryStruct.comp one (Q.d 1 0)) (Categor …
      succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (CategoryTheory.Ca …
      m : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => (P.mkHomAux Q zero one one …
    -/
    exact (mkHomAux P Q zero one one_zero_comm succ m).2.2
    /-
      🎉 no goals
    -/


@[simp]
theorem mkHom_f_0 : (mkHom P Q zero one one_zero_comm succ).f 0 = zero :=
  rfl


@[simp]
theorem mkHom_f_1 : (mkHom P Q zero one one_zero_comm succ).f 1 = one :=
  rfl


@[simp]
theorem mkHom_f_succ_succ (n : ℕ) :
    (mkHom P Q zero one one_zero_comm succ).f (n + 2) =
      (succ n
          ⟨(mkHom P Q zero one one_zero_comm succ).f n,
            (mkHom P Q zero one one_zero_comm succ).f (n + 1),
            (mkHom P Q zero one one_zero_comm succ).comm (n + 1) n⟩).1 := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    P Q : ChainComplex V Nat
    zero : Quiver.Hom (P.X 0) (Q.X 0)
    one : Quiver.Hom (P.X 1) (Q.X 1)
    one_zero_comm : Eq (CategoryTheory.CategoryStruct.comp one (Q.d 1 0)) (Categor …
    succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (CategoryTheory.Ca …
    n : Nat
    ⊢ Eq ((P.mkHom Q zero one one_zero_comm succ).f (HAdd.hAdd n 2)) (succ n ⟨(P.m …
  -/
  dsimp [mkHom, mkHomAux]
  /-
    🎉 no goals
  -/


/-- Construct an `α`-indexed cochain complex from a dependently-typed differential.
-/
def of (X : α → V) (d : ∀ n, X n ⟶ X (n + 1)) (sq : ∀ n, d n ≫ d (n + 1) = 0) :
    CochainComplex V α :=
  { X := X
                                                            /-
                                                              ι : Type u_1
                                                              V : Type u
                                                              inst✝⁴ : CategoryTheory.Category.{v, u} V
                                                              inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
                                                              α : Type u_2
                                                              inst✝² : AddRightCancelSemigroup α
                                                              inst✝¹ : One α
                                                              inst✝ : DecidableEq α
                                                              X : α → V
                                                              d : (n : α) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
                                                              sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d n) (d (HAdd.hAdd n 1 …
                                                              i j : α
                                                              h : Eq (HAdd.hAdd i 1) j
                                                              ⊢ Eq (X (HAdd.hAdd i 1)) (X j)
                                                            -/
    d := fun i j => if h : i + 1 = j then d _ ≫ eqToHom (by rw [h]) else 0
                                                            /-
                                                              🎉 no goals
                                                            -/
    shape := fun i j w => by
      /-
        ι : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
        α : Type u_2
        inst✝² : AddRightCancelSemigroup α
        inst✝¹ : One α
        inst✝ : DecidableEq α
        X : α → V
        d : (n : α) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
        sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d n) (d (HAdd.hAdd n 1 …
        i j : α
        w : Not ((ComplexShape.up α).Rel i j)
        ⊢ Eq ((fun i j => dite (Eq (HAdd.hAdd i 1) j) (fun h => CategoryTheory.Categor …
      -/
      dsimp
      /-
        ι : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
        α : Type u_2
        inst✝² : AddRightCancelSemigroup α
        inst✝¹ : One α
        inst✝ : DecidableEq α
        X : α → V
        d : (n : α) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
        sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d n) (d (HAdd.hAdd n 1 …
        i j : α
        w : Not ((ComplexShape.up α).Rel i j)
        ⊢ Eq (dite (Eq (HAdd.hAdd i 1) j) (fun h => CategoryTheory.CategoryStruct.comp …
      -/
      rw [dif_neg]
      /-
        case hnc
        ι : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
        α : Type u_2
        inst✝² : AddRightCancelSemigroup α
        inst✝¹ : One α
        inst✝ : DecidableEq α
        X : α → V
        d : (n : α) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
        sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d n) (d (HAdd.hAdd n 1 …
        i j : α
        w : Not ((ComplexShape.up α).Rel i j)
        ⊢ Not (Eq (HAdd.hAdd i 1) j)
      -/
      exact w
      /-
        🎉 no goals
      -/
    d_comp_d' := fun i j k => by
      /-
        ι : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
        α : Type u_2
        inst✝² : AddRightCancelSemigroup α
        inst✝¹ : One α
        inst✝ : DecidableEq α
        X : α → V
        d : (n : α) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
        sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d n) (d (HAdd.hAdd n 1 …
        i j k : α
        ⊢ (ComplexShape.up α).Rel i j → (ComplexShape.up α).Rel j k → Eq (CategoryTheo …
      -/
      dsimp
      /-
        ι : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
        α : Type u_2
        inst✝² : AddRightCancelSemigroup α
        inst✝¹ : One α
        inst✝ : DecidableEq α
        X : α → V
        d : (n : α) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
        sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d n) (d (HAdd.hAdd n 1 …
        i j k : α
        ⊢ Eq (HAdd.hAdd i 1) j → Eq (HAdd.hAdd j 1) k → Eq (CategoryTheory.CategoryStr …
      -/
      split_ifs with h h' h'
        /-
          case pos
          ι : Type u_1
          V : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} V
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
          α : Type u_2
          inst✝² : AddRightCancelSemigroup α
          inst✝¹ : One α
          inst✝ : DecidableEq α
          X : α → V
          d : (n : α) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
          sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d n) (d (HAdd.hAdd n 1 …
          i j k : α
          h : Eq (HAdd.hAdd i 1) j
          h' : Eq (HAdd.hAdd j 1) k
          ⊢ Eq (HAdd.hAdd i 1) j → Eq (HAdd.hAdd j 1) k → Eq (CategoryTheory.CategoryStr …
        -/
      · substs h h'
        /-
          case pos
          ι : Type u_1
          V : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} V
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
          α : Type u_2
          inst✝² : AddRightCancelSemigroup α
          inst✝¹ : One α
          inst✝ : DecidableEq α
          X : α → V
          d : (n : α) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
          sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d n) (d (HAdd.hAdd n 1 …
          i : α
          ⊢ Eq (HAdd.hAdd i 1) (HAdd.hAdd i 1) → Eq (HAdd.hAdd (HAdd.hAdd i 1) 1) (HAdd. …
        -/
        simp [sq]
        /-
          🎉 no goals
        -/
      /-
        case neg
        ι : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
        α : Type u_2
        inst✝² : AddRightCancelSemigroup α
        inst✝¹ : One α
        inst✝ : DecidableEq α
        X : α → V
        d : (n : α) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
        sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d n) (d (HAdd.hAdd n 1 …
        i j k : α
        h : Eq (HAdd.hAdd i 1) j
        h' : Not (Eq (HAdd.hAdd j 1) k)
        ⊢ Eq (HAdd.hAdd i 1) j → Eq (HAdd.hAdd j 1) k → Eq (CategoryTheory.CategoryStr …
      -/
      all_goals simp }
      /-
        🎉 no goals
      -/


@[simp]
theorem of_d (j : α) : (of X d sq).d j (j + 1) = d j := by
  /-
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
    α : Type u_2
    inst✝² : AddRightCancelSemigroup α
    inst✝¹ : One α
    inst✝ : DecidableEq α
    X : α → V
    d : (n : α) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
    sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d n) (d (HAdd.hAdd n 1 …
    j : α
    ⊢ Eq ((CochainComplex.of X d sq).d j (HAdd.hAdd j 1)) (d j)
  -/
  dsimp [of]
  /-
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
    α : Type u_2
    inst✝² : AddRightCancelSemigroup α
    inst✝¹ : One α
    inst✝ : DecidableEq α
    X : α → V
    d : (n : α) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
    sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d n) (d (HAdd.hAdd n 1 …
    j : α
    ⊢ Eq (ite (Eq (HAdd.hAdd j 1) (HAdd.hAdd j 1)) (CategoryTheory.CategoryStruct. …
  -/
  rw [if_pos rfl, Category.comp_id]
  /-
    🎉 no goals
  -/


theorem of_d_ne {i j : α} (h : i + 1 ≠ j) : (of X d sq).d i j = 0 := by
  /-
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
    α : Type u_2
    inst✝² : AddRightCancelSemigroup α
    inst✝¹ : One α
    inst✝ : DecidableEq α
    X : α → V
    d : (n : α) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
    sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d n) (d (HAdd.hAdd n 1 …
    i j : α
    h : Ne (HAdd.hAdd i 1) j
    ⊢ Eq ((CochainComplex.of X d sq).d i j) 0
  -/
  dsimp [of]
  /-
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
    α : Type u_2
    inst✝² : AddRightCancelSemigroup α
    inst✝¹ : One α
    inst✝ : DecidableEq α
    X : α → V
    d : (n : α) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
    sq : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d n) (d (HAdd.hAdd n 1 …
    i j : α
    h : Ne (HAdd.hAdd i 1) j
    ⊢ Eq (dite (Eq (HAdd.hAdd i 1) j) (fun h => CategoryTheory.CategoryStruct.comp …
  -/
  rw [dif_neg h]
  /-
    🎉 no goals
  -/


/--
A constructor for chain maps between `α`-indexed cochain complexes built using `CochainComplex.of`,
from a dependently typed collection of morphisms.
-/
@[simps]
def ofHom (f : ∀ i : α, X i ⟶ Y i) (comm : ∀ i : α, f i ≫ d_Y i = d_X i ≫ f (i + 1)) :
    of X d_X sq_X ⟶ of Y d_Y sq_Y :=
  { f
    comm' := fun n m => by
      /-
        ι : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
        α : Type u_2
        inst✝² : AddRightCancelSemigroup α
        inst✝¹ : One α
        inst✝ : DecidableEq α
        X : α → V
        d_X : (n : α) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
        sq_X : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_X n) (d_X (HAdd.hA …
        Y : α → V
        d_Y : (n : α) → Quiver.Hom (Y n) (Y (HAdd.hAdd n 1))
        sq_Y : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_Y n) (d_Y (HAdd.hA …
        f : (i : α) → Quiver.Hom (X i) (Y i)
        comm : ∀ (i : α), Eq (CategoryTheory.CategoryStruct.comp (f i) (d_Y i)) (Categ …
        n m : α
        ⊢ (ComplexShape.up α).Rel n m → Eq (CategoryTheory.CategoryStruct.comp (f n) ( …
      -/
      by_cases h : n + 1 = m
        /-
          case pos
          ι : Type u_1
          V : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} V
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
          α : Type u_2
          inst✝² : AddRightCancelSemigroup α
          inst✝¹ : One α
          inst✝ : DecidableEq α
          X : α → V
          d_X : (n : α) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
          sq_X : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_X n) (d_X (HAdd.hA …
          Y : α → V
          d_Y : (n : α) → Quiver.Hom (Y n) (Y (HAdd.hAdd n 1))
          sq_Y : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_Y n) (d_Y (HAdd.hA …
          f : (i : α) → Quiver.Hom (X i) (Y i)
          comm : ∀ (i : α), Eq (CategoryTheory.CategoryStruct.comp (f i) (d_Y i)) (Categ …
          n m : α
          h : Eq (HAdd.hAdd n 1) m
          ⊢ (ComplexShape.up α).Rel n m → Eq (CategoryTheory.CategoryStruct.comp (f n) ( …
        -/
      · subst h
        /-
          case pos
          ι : Type u_1
          V : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} V
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
          α : Type u_2
          inst✝² : AddRightCancelSemigroup α
          inst✝¹ : One α
          inst✝ : DecidableEq α
          X : α → V
          d_X : (n : α) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
          sq_X : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_X n) (d_X (HAdd.hA …
          Y : α → V
          d_Y : (n : α) → Quiver.Hom (Y n) (Y (HAdd.hAdd n 1))
          sq_Y : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_Y n) (d_Y (HAdd.hA …
          f : (i : α) → Quiver.Hom (X i) (Y i)
          comm : ∀ (i : α), Eq (CategoryTheory.CategoryStruct.comp (f i) (d_Y i)) (Categ …
          n : α
          ⊢ (ComplexShape.up α).Rel n (HAdd.hAdd n 1) → Eq (CategoryTheory.CategoryStruc …
        -/
        simpa using comm n
        /-
          🎉 no goals
        -/
        /-
          case neg
          ι : Type u_1
          V : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} V
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
          α : Type u_2
          inst✝² : AddRightCancelSemigroup α
          inst✝¹ : One α
          inst✝ : DecidableEq α
          X : α → V
          d_X : (n : α) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
          sq_X : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_X n) (d_X (HAdd.hA …
          Y : α → V
          d_Y : (n : α) → Quiver.Hom (Y n) (Y (HAdd.hAdd n 1))
          sq_Y : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_Y n) (d_Y (HAdd.hA …
          f : (i : α) → Quiver.Hom (X i) (Y i)
          comm : ∀ (i : α), Eq (CategoryTheory.CategoryStruct.comp (f i) (d_Y i)) (Categ …
          n m : α
          h : Not (Eq (HAdd.hAdd n 1) m)
          ⊢ (ComplexShape.up α).Rel n m → Eq (CategoryTheory.CategoryStruct.comp (f n) ( …
        -/
      · rw [of_d_ne X _ _ h, of_d_ne Y _ _ h]
        /-
          case neg
          ι : Type u_1
          V : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} V
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
          α : Type u_2
          inst✝² : AddRightCancelSemigroup α
          inst✝¹ : One α
          inst✝ : DecidableEq α
          X : α → V
          d_X : (n : α) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
          sq_X : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_X n) (d_X (HAdd.hA …
          Y : α → V
          d_Y : (n : α) → Quiver.Hom (Y n) (Y (HAdd.hAdd n 1))
          sq_Y : ∀ (n : α), Eq (CategoryTheory.CategoryStruct.comp (d_Y n) (d_Y (HAdd.hA …
          f : (i : α) → Quiver.Hom (X i) (Y i)
          comm : ∀ (i : α), Eq (CategoryTheory.CategoryStruct.comp (f i) (d_Y i)) (Categ …
          n m : α
          h : Not (Eq (HAdd.hAdd n 1) m)
          ⊢ (ComplexShape.up α).Rel n m → Eq (CategoryTheory.CategoryStruct.comp (f n) 0 …
        -/
        simp }
        /-
          🎉 no goals
        -/


/-- An inductive constructor for `ℕ`-indexed cochain complexes.

You provide explicitly the first two differentials,
then a function which takes two differentials and the fact they compose to zero,
and returns the next object, its differential, and the fact it composes appropriately to zero.

See also `mk'`, which only sees the previous differential in the inductive step.
-/
def mk : CochainComplex V ℕ :=
  of (fun n => (mkAux X₀ X₁ X₂ d₀ d₁ s succ n).X₁) (fun n => (mkAux X₀ X₁ X₂ d₀ d₁ s succ n).f)
    fun n => (mkAux X₀ X₁ X₂ d₀ d₁ s succ n).zero


@[simp]
theorem mk_d_1_0 : (mk X₀ X₁ X₂ d₀ d₁ s succ).d 0 1 = d₀ := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    X₀ X₁ X₂ : V
    d₀ : Quiver.Hom X₀ X₁
    d₁ : Quiver.Hom X₁ X₂
    s : Eq (CategoryTheory.CategoryStruct.comp d₀ d₁) 0
    succ : (S : CategoryTheory.ShortComplex V) → PSigma fun X₄ => PSigma fun d₂ => …
    ⊢ Eq ((CochainComplex.mk X₀ X₁ X₂ d₀ d₁ s succ).d 0 1) d₀
  -/
  change ite (1 = 0 + 1) (d₀ ≫ 𝟙 X₁) 0 = d₀
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    X₀ X₁ X₂ : V
    d₀ : Quiver.Hom X₀ X₁
    d₁ : Quiver.Hom X₁ X₂
    s : Eq (CategoryTheory.CategoryStruct.comp d₀ d₁) 0
    succ : (S : CategoryTheory.ShortComplex V) → PSigma fun X₄ => PSigma fun d₂ => …
    ⊢ Eq (ite (Eq 1 (HAdd.hAdd 0 1)) (CategoryTheory.CategoryStruct.comp d₀ (Categ …
  -/
  rw [if_pos rfl, Category.comp_id]
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_d_2_0 : (mk X₀ X₁ X₂ d₀ d₁ s succ).d 1 2 = d₁ := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    X₀ X₁ X₂ : V
    d₀ : Quiver.Hom X₀ X₁
    d₁ : Quiver.Hom X₁ X₂
    s : Eq (CategoryTheory.CategoryStruct.comp d₀ d₁) 0
    succ : (S : CategoryTheory.ShortComplex V) → PSigma fun X₄ => PSigma fun d₂ => …
    ⊢ Eq ((CochainComplex.mk X₀ X₁ X₂ d₀ d₁ s succ).d 1 2) d₁
  -/
  change ite (2 = 1 + 1) (d₁ ≫ 𝟙 X₂) 0 = d₁
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    X₀ X₁ X₂ : V
    d₀ : Quiver.Hom X₀ X₁
    d₁ : Quiver.Hom X₁ X₂
    s : Eq (CategoryTheory.CategoryStruct.comp d₀ d₁) 0
    succ : (S : CategoryTheory.ShortComplex V) → PSigma fun X₄ => PSigma fun d₂ => …
    ⊢ Eq (ite (Eq 2 (HAdd.hAdd 1 1)) (CategoryTheory.CategoryStruct.comp d₁ (Categ …
  -/
  rw [if_pos rfl, Category.comp_id]
  /-
    🎉 no goals
  -/

-- TODO simp lemmas for the inductive steps? It's not entirely clear that they are needed.

/-- A simpler inductive constructor for `ℕ`-indexed cochain complexes.

You provide explicitly the first differential,
then a function which takes a differential,
and returns the next object, its differential, and the fact it composes appropriately to zero.
-/
def mk' (X₀ X₁ : V) (d : X₀ ⟶ X₁)
    -- (succ' : ∀ : ΣX₀ X₁ : V, X₀ ⟶ X₁, Σ' (X₂ : V) (d : t.2.1 ⟶ X₂), t.2.2 ≫ d = 0) :
    (succ' : ∀ {X₀ X₁ : V} (f : X₀ ⟶ X₁), Σ' (X₂ : V) (d : X₁ ⟶ X₂), f ≫ d = 0) :
    CochainComplex V ℕ :=
  mk _ _ _ _ _ (succ' d).2.2 (fun S => succ' S.g)


@[simp]
theorem mk'_X_1 : (mk' X₀ X₁ d₀ succ').X 1 = X₁ :=
  rfl


@[simp]
theorem mk'_d_1_0 : (mk' X₀ X₁ d₀ succ').d 0 1 = d₀ := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    X₀ X₁ : V
    d₀ : Quiver.Hom X₀ X₁
    succ' : {X₀ X₁ : V} → (f : Quiver.Hom X₀ X₁) → PSigma fun X₂ => PSigma fun d = …
    ⊢ Eq ((CochainComplex.mk' X₀ X₁ d₀ fun {X₀ X₁} => succ').d 0 1) d₀
  -/
  change ite (1 = 0 + 1) (d₀ ≫ 𝟙 X₁) 0 = d₀
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    X₀ X₁ : V
    d₀ : Quiver.Hom X₀ X₁
    succ' : {X₀ X₁ : V} → (f : Quiver.Hom X₀ X₁) → PSigma fun X₂ => PSigma fun d = …
    ⊢ Eq (ite (Eq 1 (HAdd.hAdd 0 1)) (CategoryTheory.CategoryStruct.comp d₀ (Categ …
  -/
  rw [if_pos rfl, Category.comp_id]
  /-
    🎉 no goals
  -/

-- TODO simp lemmas for the inductive steps? It's not entirely clear that they are needed.

/-- An auxiliary construction for `mkHom`.

Here we build by induction a family of commutative squares,
but don't require at the type level that these successive commutative squares actually agree.
They do in fact agree, and we then capture that at the type level (i.e. by constructing a chain map)
in `mkHom`.
-/
def mkHomAux :
    ∀ n,
      Σ' (f : P.X n ⟶ Q.X n) (f' : P.X (n + 1) ⟶ Q.X (n + 1)),
        f ≫ Q.d n (n + 1) = P.d n (n + 1) ≫ f'
  | 0 => ⟨zero, one, one_zero_comm⟩
  | n + 1 => ⟨(mkHomAux n).2.1, (succ n (mkHomAux n)).1, (succ n (mkHomAux n)).2⟩


/-- A constructor for chain maps between `ℕ`-indexed cochain complexes,
working by induction on commutative squares.

You need to provide the components of the chain map in degrees 0 and 1,
show that these form a commutative square,
and then give a construction of each component,
and the fact that it forms a commutative square with the previous component,
using as an inductive hypothesis the data (and commutativity) of the previous two components.
-/
def mkHom : P ⟶ Q where
  f n := (mkHomAux P Q zero one one_zero_comm succ n).1
  comm' n m := by
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      P Q : CochainComplex V Nat
      zero : Quiver.Hom (P.X 0) (Q.X 0)
      one : Quiver.Hom (P.X 1) (Q.X 1)
      one_zero_comm : Eq (CategoryTheory.CategoryStruct.comp zero (Q.d 0 1)) (Catego …
      succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (CategoryTheory.Ca …
      n m : Nat
      ⊢ (ComplexShape.up Nat).Rel n m → Eq (CategoryTheory.CategoryStruct.comp ((fun …
    -/
    rintro (rfl : n + 1 = m)
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      P Q : CochainComplex V Nat
      zero : Quiver.Hom (P.X 0) (Q.X 0)
      one : Quiver.Hom (P.X 1) (Q.X 1)
      one_zero_comm : Eq (CategoryTheory.CategoryStruct.comp zero (Q.d 0 1)) (Catego …
      succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (CategoryTheory.Ca …
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => (P.mkHomAux Q zero one one …
    -/
    exact (mkHomAux P Q zero one one_zero_comm succ n).2.2
    /-
      🎉 no goals
    -/


@[simp]
theorem mkHom_f_succ_succ (n : ℕ) :
    (mkHom P Q zero one one_zero_comm succ).f (n + 2) =
      (succ n
          ⟨(mkHom P Q zero one one_zero_comm succ).f n,
            (mkHom P Q zero one one_zero_comm succ).f (n + 1),
            (mkHom P Q zero one one_zero_comm succ).comm n (n + 1)⟩).1 := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    P Q : CochainComplex V Nat
    zero : Quiver.Hom (P.X 0) (Q.X 0)
    one : Quiver.Hom (P.X 1) (Q.X 1)
    one_zero_comm : Eq (CategoryTheory.CategoryStruct.comp zero (Q.d 0 1)) (Catego …
    succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (CategoryTheory.Ca …
    n : Nat
    ⊢ Eq ((P.mkHom Q zero one one_zero_comm succ).f (HAdd.hAdd n 2)) (succ n ⟨(P.m …
  -/
  dsimp [mkHom, mkHomAux]
  /-
    🎉 no goals
  -/



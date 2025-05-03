/-- A term of type `HomComplex.Triplet n` consists of two integers `p` and `q`
such that `p + n = q`. (This type is introduced so that the instance
`AddCommGroup (Cochain F G n)` defined below can be found automatically.) -/
structure Triplet (n : ℤ) where
  /-- a first integer -/
  p : ℤ
  /-- a second integer -/
  q : ℤ
  /-- the condition on the two integers -/
  hpq : p + n = q


/-- A cochain of degree `n : ℤ` between to cochain complexes `F` and `G` consists
of a family of morphisms `F.X p ⟶ G.X q` whenever `p + n = q`, i.e. for all
triplets in `HomComplex.Triplet n`. -/
def Cochain := ∀ (T : Triplet n), F.X T.p ⟶ G.X T.q


instance : AddCommGroup (Cochain F G n) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    F G K L : CochainComplex C Int
    n m : Int
    ⊢ AddCommGroup (CochainComplex.HomComplex.Cochain F G n)
  -/
  dsimp only [Cochain]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    F G K L : CochainComplex C Int
    n m : Int
    ⊢ AddCommGroup ((T : CochainComplex.HomComplex.Triplet n) → Quiver.Hom (F.X T. …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : Module R (Cochain F G n) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    F G K L : CochainComplex C Int
    n m : Int
    ⊢ Module R (CochainComplex.HomComplex.Cochain F G n)
  -/
  dsimp only [Cochain]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    F G K L : CochainComplex C Int
    n m : Int
    ⊢ Module R ((T : CochainComplex.HomComplex.Triplet n) → Quiver.Hom (F.X T.p) ( …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- A practical constructor for cochains. -/
def mk (v : ∀ (p q : ℤ) (_ : p + n = q), F.X p ⟶ G.X q) : Cochain F G n :=
  fun ⟨p, q, hpq⟩ => v p q hpq


/-- The value of a cochain on a triplet `⟨p, q, hpq⟩`. -/
def v (γ : Cochain F G n) (p q : ℤ) (hpq : p + n = q) :
    F.X p ⟶ G.X q := γ ⟨p, q, hpq⟩


@[simp]
lemma mk_v (v : ∀ (p q : ℤ) (_ : p + n = q), F.X p ⟶ G.X q) (p q : ℤ) (hpq : p + n = q) :
    (Cochain.mk v).v p q hpq = v p q hpq := rfl


lemma congr_v {z₁ z₂ : Cochain F G n} (h : z₁ = z₂) (p q : ℤ) (hpq : p + n = q) :
                                      /-
                                        C : Type u
                                        inst✝¹ : CategoryTheory.Category.{v, u} C
                                        inst✝ : CategoryTheory.Preadditive C
                                        F G : CochainComplex C Int
                                        n : Int
                                        z₁ z₂ : CochainComplex.HomComplex.Cochain F G n
                                        h : Eq z₁ z₂
                                        p q : Int
                                        hpq : Eq (HAdd.hAdd p n) q
                                        ⊢ Eq (z₁.v p q hpq) (z₂.v p q hpq)
                                      -/
    z₁.v p q hpq = z₂.v p q hpq := by subst h; rfl
                                               /-
                                                 🎉 no goals
                                               -/


@[ext]
lemma ext (z₁ z₂ : Cochain F G n)
    (h : ∀ (p q hpq), z₁.v p q hpq = z₂.v p q hpq) : z₁ = z₂ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n : Int
    z₁ z₂ : CochainComplex.HomComplex.Cochain F G n
    h : ∀ (p q : Int) (hpq : Eq (HAdd.hAdd p n) q), Eq (z₁.v p q hpq) (z₂.v p q hpq)
    ⊢ Eq z₁ z₂
  -/
  funext ⟨p, q, hpq⟩
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n : Int
    z₁ z₂ : CochainComplex.HomComplex.Cochain F G n
    h : ∀ (p q : Int) (hpq : Eq (HAdd.hAdd p n) q), Eq (z₁.v p q hpq) (z₂.v p q hpq)
    p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    ⊢ Eq (z₁ { p := p, q := q, hpq := hpq }) (z₂ { p := p, q := q, hpq := hpq })
  -/
  apply h
  /-
    🎉 no goals
  -/


@[ext 1100]
lemma ext₀ (z₁ z₂ : Cochain F G 0)
    (h : ∀ (p : ℤ), z₁.v p p (add_zero p) = z₂.v p p (add_zero p)) : z₁ = z₂ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    z₁ z₂ : CochainComplex.HomComplex.Cochain F G 0
    h : ∀ (p : Int), Eq (z₁.v p p ⋯) (z₂.v p p ⋯)
    ⊢ Eq z₁ z₂
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    z₁ z₂ : CochainComplex.HomComplex.Cochain F G 0
    h : ∀ (p : Int), Eq (z₁.v p p ⋯) (z₂.v p p ⋯)
    p q : Int
    hpq : Eq (HAdd.hAdd p 0) q
    ⊢ Eq (z₁.v p q hpq) (z₂.v p q hpq)
  -/
  obtain rfl : q = p := by rw [← hpq, add_zero]
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    z₁ z₂ : CochainComplex.HomComplex.Cochain F G 0
    h : ∀ (p : Int), Eq (z₁.v p p ⋯) (z₂.v p p ⋯)
    q : Int
    hpq : Eq (HAdd.hAdd q 0) q
    ⊢ Eq (z₁.v q q hpq) (z₂.v q q hpq)
  -/
  exact h q
  /-
    🎉 no goals
  -/


@[simp]
lemma zero_v {n : ℤ} (p q : ℤ) (hpq : p + n = q) :
    (0 : Cochain F G n).v p q hpq = 0 := rfl


@[simp]
lemma add_v {n : ℤ} (z₁ z₂ : Cochain F G n) (p q : ℤ) (hpq : p + n = q) :
    (z₁ + z₂).v p q hpq = z₁.v p q hpq + z₂.v p q hpq := rfl


@[simp]
lemma sub_v {n : ℤ} (z₁ z₂ : Cochain F G n) (p q : ℤ) (hpq : p + n = q) :
    (z₁ - z₂).v p q hpq = z₁.v p q hpq - z₂.v p q hpq := rfl


@[simp]
lemma neg_v {n : ℤ} (z : Cochain F G n) (p q : ℤ) (hpq : p + n = q) :
    (-z).v p q hpq = - (z.v p q hpq) := rfl


@[simp]
lemma smul_v {n : ℤ} (k : R) (z : Cochain F G n) (p q : ℤ) (hpq : p + n = q) :
    (k • z).v p q hpq = k • (z.v p q hpq) := rfl


@[simp]
lemma units_smul_v {n : ℤ} (k : Rˣ) (z : Cochain F G n) (p q : ℤ) (hpq : p + n = q) :
    (k • z).v p q hpq = k • (z.v p q hpq) := rfl


/-- A cochain of degree `0` from `F` to `G` can be constructed from a family
of morphisms `F.X p ⟶ G.X p` for all `p : ℤ`. -/
def ofHoms (ψ : ∀ (p : ℤ), F.X p ⟶ G.X p) : Cochain F G 0 :=
                                               /-
                                                 C : Type u
                                                 inst✝³ : CategoryTheory.Category.{v, u} C
                                                 inst✝² : CategoryTheory.Preadditive C
                                                 R : Type u_1
                                                 inst✝¹ : Ring R
                                                 inst✝ : CategoryTheory.Linear R C
                                                 F G K L : CochainComplex C Int
                                                 n m : Int
                                                 ψ : (p : Int) → Quiver.Hom (F.X p) (G.X p)
                                                 p q : Int
                                                 hpq : Eq (HAdd.hAdd p 0) q
                                                 ⊢ Eq (G.X p) (G.X q)
                                               -/
  Cochain.mk (fun p q hpq => ψ p ≫ eqToHom (by rw [← hpq, add_zero]))
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
lemma ofHoms_v (ψ : ∀ (p : ℤ), F.X p ⟶ G.X p) (p : ℤ) :
    (ofHoms ψ).v p p (add_zero p) = ψ p := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    ψ : (p : Int) → Quiver.Hom (F.X p) (G.X p)
    p : Int
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.ofHoms ψ).v p p ⋯) (ψ p)
  -/
  simp only [ofHoms, mk_v, eqToHom_refl, comp_id]
  /-
    🎉 no goals
  -/


@[simp]
                                                                    /-
                                                                      C : Type u
                                                                      inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                      inst✝ : CategoryTheory.Preadditive C
                                                                      F G : CochainComplex C Int
                                                                      ⊢ Eq (CochainComplex.HomComplex.Cochain.ofHoms fun p => 0) 0
                                                                    -/
lemma ofHoms_zero : ofHoms (fun p => (0 : F.X p ⟶ G.X p)) = 0 := by aesop_cat
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
lemma ofHoms_v_comp_d (ψ : ∀ (p : ℤ), F.X p ⟶ G.X p) (p q q' : ℤ) (hpq : p + 0 = q) :
    (ofHoms ψ).v p q hpq ≫ G.d q q' = ψ p ≫ G.d p q' := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    ψ : (p : Int) → Quiver.Hom (F.X p) (G.X p)
    p q q' : Int
    hpq : Eq (HAdd.hAdd p 0) q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.HomComplex.Cochain.o …
  -/
  rw [add_zero] at hpq
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    ψ : (p : Int) → Quiver.Hom (F.X p) (G.X p)
    p q q' : Int
    hpq✝ : Eq (HAdd.hAdd p 0) q
    hpq : Eq p q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.HomComplex.Cochain.o …
  -/
  subst hpq
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    ψ : (p : Int) → Quiver.Hom (F.X p) (G.X p)
    p q' : Int
    hpq : Eq (HAdd.hAdd p 0) p
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.HomComplex.Cochain.o …
  -/
  rw [ofHoms_v]
  /-
    🎉 no goals
  -/


@[simp]
lemma d_comp_ofHoms_v (ψ : ∀ (p : ℤ), F.X p ⟶ G.X p) (p' p q : ℤ) (hpq : p + 0 = q) :
    F.d p' p ≫ (ofHoms ψ).v p q hpq = F.d p' q ≫ ψ q := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    ψ : (p : Int) → Quiver.Hom (F.X p) (G.X p)
    p' p q : Int
    hpq : Eq (HAdd.hAdd p 0) q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.d p' p) ((CochainComplex.HomComple …
  -/
  rw [add_zero] at hpq
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    ψ : (p : Int) → Quiver.Hom (F.X p) (G.X p)
    p' p q : Int
    hpq✝ : Eq (HAdd.hAdd p 0) q
    hpq : Eq p q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.d p' p) ((CochainComplex.HomComple …
  -/
  subst hpq
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    ψ : (p : Int) → Quiver.Hom (F.X p) (G.X p)
    p' p : Int
    hpq : Eq (HAdd.hAdd p 0) p
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.d p' p) ((CochainComplex.HomComple …
  -/
  rw [ofHoms_v]
  /-
    🎉 no goals
  -/


/-- The `0`-cochain attached to a morphism of cochain complexes. -/
def ofHom (φ : F ⟶ G) : Cochain F G 0 := ofHoms (fun p => φ.f p)


@[simp]
lemma ofHom_zero : ofHom (0 : F ⟶ G) = 0 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    ⊢ Eq (CochainComplex.HomComplex.Cochain.ofHom 0) 0
  -/
  simp only [ofHom, HomologicalComplex.zero_f_apply, ofHoms_zero]
  /-
    🎉 no goals
  -/


@[simp]
lemma ofHom_v (φ : F ⟶ G) (p : ℤ) : (ofHom φ).v p p (add_zero p) = φ.f p := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    p : Int
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.ofHom φ).v p p ⋯) (φ.f p)
  -/
  simp only [ofHom, ofHoms_v]
  /-
    🎉 no goals
  -/


@[simp]
lemma ofHom_v_comp_d (φ : F ⟶ G) (p q q' : ℤ) (hpq : p + 0 = q) :
    (ofHom φ).v p q hpq ≫ G.d q q' = φ.f p ≫ G.d p q' := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    p q q' : Int
    hpq : Eq (HAdd.hAdd p 0) q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.HomComplex.Cochain.o …
  -/
  simp only [ofHom, ofHoms_v_comp_d]
  /-
    🎉 no goals
  -/


@[simp]
lemma d_comp_ofHom_v (φ : F ⟶ G) (p' p q : ℤ) (hpq : p + 0 = q) :
    F.d p' p ≫ (ofHom φ).v p q hpq = F.d p' q ≫ φ.f q := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    p' p q : Int
    hpq : Eq (HAdd.hAdd p 0) q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.d p' p) ((CochainComplex.HomComple …
  -/
  simp only [ofHom, d_comp_ofHoms_v]
  /-
    🎉 no goals
  -/


@[simp]
lemma ofHom_add (φ₁ φ₂ : F ⟶ G) :
                                                                        /-
                                                                          C : Type u
                                                                          inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                          inst✝ : CategoryTheory.Preadditive C
                                                                          F G : CochainComplex C Int
                                                                          φ₁ φ₂ : Quiver.Hom F G
                                                                          ⊢ Eq (CochainComplex.HomComplex.Cochain.ofHom (HAdd.hAdd φ₁ φ₂)) (HAdd.hAdd (C …
                                                                        -/
    Cochain.ofHom (φ₁ + φ₂) = Cochain.ofHom φ₁ + Cochain.ofHom φ₂ := by aesop_cat
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
lemma ofHom_sub (φ₁ φ₂ : F ⟶ G) :
                                                                        /-
                                                                          C : Type u
                                                                          inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                          inst✝ : CategoryTheory.Preadditive C
                                                                          F G : CochainComplex C Int
                                                                          φ₁ φ₂ : Quiver.Hom F G
                                                                          ⊢ Eq (CochainComplex.HomComplex.Cochain.ofHom (HSub.hSub φ₁ φ₂)) (HSub.hSub (C …
                                                                        -/
    Cochain.ofHom (φ₁ - φ₂) = Cochain.ofHom φ₁ - Cochain.ofHom φ₂ := by aesop_cat
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
lemma ofHom_neg (φ : F ⟶ G) :
                                                /-
                                                  C : Type u
                                                  inst✝¹ : CategoryTheory.Category.{v, u} C
                                                  inst✝ : CategoryTheory.Preadditive C
                                                  F G : CochainComplex C Int
                                                  φ : Quiver.Hom F G
                                                  ⊢ Eq (CochainComplex.HomComplex.Cochain.ofHom (Neg.neg φ)) (Neg.neg (CochainCo …
                                                -/
    Cochain.ofHom (-φ) = -Cochain.ofHom φ := by aesop_cat
                                                /-
                                                  🎉 no goals
                                                -/


/-- The cochain of degree `-1` given by an homotopy between two morphism of complexes. -/
def ofHomotopy {φ₁ φ₂ : F ⟶ G} (ho : Homotopy φ₁ φ₂) : Cochain F G (-1) :=
  Cochain.mk (fun p q _ => ho.hom p q)


@[simp]
lemma ofHomotopy_ofEq {φ₁ φ₂ : F ⟶ G} (h : φ₁ = φ₂) :
    ofHomotopy (Homotopy.ofEq h) = 0 := rfl


@[simp]
lemma ofHomotopy_refl (φ : F ⟶ G) :
    ofHomotopy (Homotopy.refl φ) = 0 := rfl


@[reassoc]
lemma v_comp_XIsoOfEq_hom
    (γ : Cochain F G n) (p q q' : ℤ) (hpq : p + n = q) (hq' : q = q') :
                                                                         /-
                                                                           C : Type u
                                                                           inst✝³ : CategoryTheory.Category.{v, u} C
                                                                           inst✝² : CategoryTheory.Preadditive C
                                                                           R : Type u_1
                                                                           inst✝¹ : Ring R
                                                                           inst✝ : CategoryTheory.Linear R C
                                                                           F G K L : CochainComplex C Int
                                                                           n m : Int
                                                                           γ : CochainComplex.HomComplex.Cochain F G n
                                                                           p q q' : Int
                                                                           hpq : Eq (HAdd.hAdd p n) q
                                                                           hq' : Eq q q'
                                                                           ⊢ Eq (HAdd.hAdd p n) q'
                                                                         -/
    γ.v p q hpq ≫ (HomologicalComplex.XIsoOfEq G hq').hom = γ.v p q' (by rw [← hq', hpq]) := by
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain F G n
    p q q' : Int
    hpq : Eq (HAdd.hAdd p n) q
    hq' : Eq q q'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (γ.v p q hpq) (HomologicalComplex.XIs …
  -/
  subst hq'
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain F G n
    p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (γ.v p q hpq) (HomologicalComplex.XIs …
  -/
  simp only [HomologicalComplex.XIsoOfEq, eqToIso_refl, Iso.refl_hom, comp_id]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma v_comp_XIsoOfEq_inv
    (γ : Cochain F G n) (p q q' : ℤ) (hpq : p + n = q) (hq' : q' = q) :
                                                                         /-
                                                                           C : Type u
                                                                           inst✝³ : CategoryTheory.Category.{v, u} C
                                                                           inst✝² : CategoryTheory.Preadditive C
                                                                           R : Type u_1
                                                                           inst✝¹ : Ring R
                                                                           inst✝ : CategoryTheory.Linear R C
                                                                           F G K L : CochainComplex C Int
                                                                           n m : Int
                                                                           γ : CochainComplex.HomComplex.Cochain F G n
                                                                           p q q' : Int
                                                                           hpq : Eq (HAdd.hAdd p n) q
                                                                           hq' : Eq q' q
                                                                           ⊢ Eq (HAdd.hAdd p n) q'
                                                                         -/
    γ.v p q hpq ≫ (HomologicalComplex.XIsoOfEq G hq').inv = γ.v p q' (by rw [hq', hpq]) := by
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain F G n
    p q q' : Int
    hpq : Eq (HAdd.hAdd p n) q
    hq' : Eq q' q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (γ.v p q hpq) (HomologicalComplex.XIs …
  -/
  subst hq'
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n : Int
    γ : CochainComplex.HomComplex.Cochain F G n
    p q' : Int
    hpq : Eq (HAdd.hAdd p n) q'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (γ.v p q' hpq) (HomologicalComplex.XI …
  -/
  simp only [HomologicalComplex.XIsoOfEq, eqToIso_refl, Iso.refl_inv, comp_id]
  /-
    🎉 no goals
  -/


/-- The composition of cochains. -/
def comp {n₁ n₂ n₁₂ : ℤ} (z₁ : Cochain F G n₁) (z₂ : Cochain G K n₂) (h : n₁ + n₂ = n₁₂) :
    Cochain F K n₁₂ :=
                                                                       /-
                                                                         C : Type u
                                                                         inst✝³ : CategoryTheory.Category.{v, u} C
                                                                         inst✝² : CategoryTheory.Preadditive C
                                                                         R : Type u_1
                                                                         inst✝¹ : Ring R
                                                                         inst✝ : CategoryTheory.Linear R C
                                                                         F G K L : CochainComplex C Int
                                                                         n m n₁ n₂ n₁₂ : Int
                                                                         z₁ : CochainComplex.HomComplex.Cochain F G n₁
                                                                         z₂ : CochainComplex.HomComplex.Cochain G K n₂
                                                                         h : Eq (HAdd.hAdd n₁ n₂) n₁₂
                                                                         p q : Int
                                                                         hpq : Eq (HAdd.hAdd p n₁₂) q
                                                                         ⊢ Eq (HAdd.hAdd (HAdd.hAdd p n₁) n₂) q
                                                                       -/
  Cochain.mk (fun p q hpq => z₁.v p (p + n₁) rfl ≫ z₂.v (p + n₁) q (by omega))
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


lemma comp_v {n₁ n₂ n₁₂ : ℤ} (z₁ : Cochain F G n₁) (z₂ : Cochain G K n₂) (h : n₁ + n₂ = n₁₂)
    (p₁ p₂ p₃ : ℤ) (h₁ : p₁ + n₁ = p₂) (h₂ : p₂ + n₂ = p₃) :
                               /-
                                 C : Type u
                                 inst✝³ : CategoryTheory.Category.{v, u} C
                                 inst✝² : CategoryTheory.Preadditive C
                                 R : Type u_1
                                 inst✝¹ : Ring R
                                 inst✝ : CategoryTheory.Linear R C
                                 F G K L : CochainComplex C Int
                                 n m n₁ n₂ n₁₂ : Int
                                 z₁ : CochainComplex.HomComplex.Cochain F G n₁
                                 z₂ : CochainComplex.HomComplex.Cochain G K n₂
                                 h : Eq (HAdd.hAdd n₁ n₂) n₁₂
                                 p₁ p₂ p₃ : Int
                                 h₁ : Eq (HAdd.hAdd p₁ n₁) p₂
                                 h₂ : Eq (HAdd.hAdd p₂ n₂) p₃
                                 ⊢ Eq (HAdd.hAdd p₁ n₁₂) p₃
                               -/
    (z₁.comp z₂ h).v p₁ p₃ (by rw [← h₂, ← h₁, ← h, add_assoc]) =
                               /-
                                 🎉 no goals
                               -/
      z₁.v p₁ p₂ h₁ ≫ z₂.v p₂ p₃ h₂ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    p₁ p₂ p₃ : Int
    h₁ : Eq (HAdd.hAdd p₁ n₁) p₂
    h₂ : Eq (HAdd.hAdd p₂ n₂) p₃
    ⊢ Eq ((z₁.comp z₂ h).v p₁ p₃ ⋯) (CategoryTheory.CategoryStruct.comp (z₁.v p₁ p …
  -/
  subst h₁; rfl
            /-
              🎉 no goals
            -/


@[simp]
lemma comp_zero_cochain_v (z₁ : Cochain F G n) (z₂ : Cochain G K 0) (p q : ℤ) (hpq : p + n = q) :
    (z₁.comp z₂ (add_zero n)).v p q hpq = z₁.v p q hpq ≫ z₂.v q q (add_zero q) :=
  comp_v z₁ z₂ (add_zero n) p q q hpq (add_zero q)


@[simp]
lemma zero_cochain_comp_v (z₁ : Cochain F G 0) (z₂ : Cochain G K n) (p q : ℤ) (hpq : p + n = q) :
    (z₁.comp z₂ (zero_add n)).v p q hpq = z₁.v p p (add_zero p) ≫ z₂.v p q hpq :=
  comp_v z₁ z₂ (zero_add n) p p q (add_zero p) hpq


/-- The associativity of the composition of cochains. -/
lemma comp_assoc {n₁ n₂ n₃ n₁₂ n₂₃ n₁₂₃ : ℤ}
    (z₁ : Cochain F G n₁) (z₂ : Cochain G K n₂) (z₃ : Cochain K L n₃)
    (h₁₂ : n₁ + n₂ = n₁₂) (h₂₃ : n₂ + n₃ = n₂₃) (h₁₂₃ : n₁ + n₂ + n₃ = n₁₂₃) :
                                                      /-
                                                        C : Type u
                                                        inst✝³ : CategoryTheory.Category.{v, u} C
                                                        inst✝² : CategoryTheory.Preadditive C
                                                        R : Type u_1
                                                        inst✝¹ : Ring R
                                                        inst✝ : CategoryTheory.Linear R C
                                                        F G K L : CochainComplex C Int
                                                        n m n₁ n₂ n₃ n₁₂ n₂₃ n₁₂₃ : Int
                                                        z₁ : CochainComplex.HomComplex.Cochain F G n₁
                                                        z₂ : CochainComplex.HomComplex.Cochain G K n₂
                                                        z₃ : CochainComplex.HomComplex.Cochain K L n₃
                                                        h₁₂ : Eq (HAdd.hAdd n₁ n₂) n₁₂
                                                        h₂₃ : Eq (HAdd.hAdd n₂ n₃) n₂₃
                                                        h₁₂₃ : Eq (HAdd.hAdd (HAdd.hAdd n₁ n₂) n₃) n₁₂₃
                                                        ⊢ Eq (HAdd.hAdd n₁₂ n₃) n₁₂₃
                                                      -/
    (z₁.comp z₂ h₁₂).comp z₃ (show n₁₂ + n₃ = n₁₂₃ by rw [← h₁₂, h₁₂₃]) =
                                                      /-
                                                        🎉 no goals
                                                      -/
                                   /-
                                     C : Type u
                                     inst✝³ : CategoryTheory.Category.{v, u} C
                                     inst✝² : CategoryTheory.Preadditive C
                                     R : Type u_1
                                     inst✝¹ : Ring R
                                     inst✝ : CategoryTheory.Linear R C
                                     F G K L : CochainComplex C Int
                                     n m n₁ n₂ n₃ n₁₂ n₂₃ n₁₂₃ : Int
                                     z₁ : CochainComplex.HomComplex.Cochain F G n₁
                                     z₂ : CochainComplex.HomComplex.Cochain G K n₂
                                     z₃ : CochainComplex.HomComplex.Cochain K L n₃
                                     h₁₂ : Eq (HAdd.hAdd n₁ n₂) n₁₂
                                     h₂₃ : Eq (HAdd.hAdd n₂ n₃) n₂₃
                                     h₁₂₃ : Eq (HAdd.hAdd (HAdd.hAdd n₁ n₂) n₃) n₁₂₃
                                     ⊢ Eq (HAdd.hAdd n₁ n₂₃) n₁₂₃
                                   -/
      z₁.comp (z₂.comp z₃ h₂₃) (by rw [← h₂₃, ← h₁₂₃, add_assoc]) := by
                                   /-
                                     🎉 no goals
                                   -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K L : CochainComplex C Int
    n₁ n₂ n₃ n₁₂ n₂₃ n₁₂₃ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    z₃ : CochainComplex.HomComplex.Cochain K L n₃
    h₁₂ : Eq (HAdd.hAdd n₁ n₂) n₁₂
    h₂₃ : Eq (HAdd.hAdd n₂ n₃) n₂₃
    h₁₂₃ : Eq (HAdd.hAdd (HAdd.hAdd n₁ n₂) n₃) n₁₂₃
    ⊢ Eq ((z₁.comp z₂ h₁₂).comp z₃ ⋯) (z₁.comp (z₂.comp z₃ h₂₃) ⋯)
  -/
  substs h₁₂ h₂₃ h₁₂₃
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K L : CochainComplex C Int
    n₁ n₂ n₃ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    z₃ : CochainComplex.HomComplex.Cochain K L n₃
    ⊢ Eq ((z₁.comp z₂ ⋯).comp z₃ ⋯) (z₁.comp (z₂.comp z₃ ⋯) ⋯)
  -/
  ext p q hpq
  rw [comp_v _ _ rfl p (p + n₁ + n₂) q (add_assoc _ _ _).symm (by omega),
    comp_v z₁ z₂ rfl p (p + n₁) (p + n₁ + n₂) (by omega) (by omega),
    comp_v z₁ (z₂.comp z₃ rfl) (add_assoc n₁ n₂ n₃).symm p (p + n₁) q (by omega) (by omega),
    comp_v z₂ z₃ rfl (p + n₁) (p + n₁ + n₂) q (by omega) (by omega), assoc]


@[simp]
lemma comp_assoc_of_first_is_zero_cochain {n₂ n₃ n₂₃ : ℤ}
    (z₁ : Cochain F G 0) (z₂ : Cochain G K n₂) (z₃ : Cochain K L n₃)
    (h₂₃ : n₂ + n₃ = n₂₃) :
    (z₁.comp z₂ (zero_add n₂)).comp z₃ h₂₃ = z₁.comp (z₂.comp z₃ h₂₃) (zero_add n₂₃) :=
                           /-
                             C : Type u
                             inst✝¹ : CategoryTheory.Category.{v, u} C
                             inst✝ : CategoryTheory.Preadditive C
                             F G K L : CochainComplex C Int
                             n₂ n₃ n₂₃ : Int
                             z₁ : CochainComplex.HomComplex.Cochain F G 0
                             z₂ : CochainComplex.HomComplex.Cochain G K n₂
                             z₃ : CochainComplex.HomComplex.Cochain K L n₃
                             h₂₃ : Eq (HAdd.hAdd n₂ n₃) n₂₃
                             ⊢ Eq (HAdd.hAdd (HAdd.hAdd 0 n₂) n₃) n₂₃
                           -/
  comp_assoc _ _ _ _ _ (by omega)
                           /-
                             🎉 no goals
                           -/


@[simp]
lemma comp_assoc_of_second_is_zero_cochain {n₁ n₃ n₁₃ : ℤ}
    (z₁ : Cochain F G n₁) (z₂ : Cochain G K 0) (z₃ : Cochain K L n₃) (h₁₃ : n₁ + n₃ = n₁₃) :
    (z₁.comp z₂ (add_zero n₁)).comp z₃ h₁₃ = z₁.comp (z₂.comp z₃ (zero_add n₃)) h₁₃ :=
                           /-
                             C : Type u
                             inst✝¹ : CategoryTheory.Category.{v, u} C
                             inst✝ : CategoryTheory.Preadditive C
                             F G K L : CochainComplex C Int
                             n₁ n₃ n₁₃ : Int
                             z₁ : CochainComplex.HomComplex.Cochain F G n₁
                             z₂ : CochainComplex.HomComplex.Cochain G K 0
                             z₃ : CochainComplex.HomComplex.Cochain K L n₃
                             h₁₃ : Eq (HAdd.hAdd n₁ n₃) n₁₃
                             ⊢ Eq (HAdd.hAdd (HAdd.hAdd n₁ 0) n₃) n₁₃
                           -/
  comp_assoc _ _ _ _ _ (by omega)
                           /-
                             🎉 no goals
                           -/


@[simp]
lemma comp_assoc_of_third_is_zero_cochain {n₁ n₂ n₁₂ : ℤ}
    (z₁ : Cochain F G n₁) (z₂ : Cochain G K n₂) (z₃ : Cochain K L 0) (h₁₂ : n₁ + n₂ = n₁₂) :
    (z₁.comp z₂ h₁₂).comp z₃ (add_zero n₁₂) = z₁.comp (z₂.comp z₃ (add_zero n₂)) h₁₂ :=
                           /-
                             C : Type u
                             inst✝¹ : CategoryTheory.Category.{v, u} C
                             inst✝ : CategoryTheory.Preadditive C
                             F G K L : CochainComplex C Int
                             n₁ n₂ n₁₂ : Int
                             z₁ : CochainComplex.HomComplex.Cochain F G n₁
                             z₂ : CochainComplex.HomComplex.Cochain G K n₂
                             z₃ : CochainComplex.HomComplex.Cochain K L 0
                             h₁₂ : Eq (HAdd.hAdd n₁ n₂) n₁₂
                             ⊢ Eq (HAdd.hAdd (HAdd.hAdd n₁ n₂) 0) n₁₂
                           -/
  comp_assoc _ _ _ _ _ (by omega)
                           /-
                             🎉 no goals
                           -/


@[simp]
lemma comp_assoc_of_second_degree_eq_neg_third_degree {n₁ n₂ n₁₂ : ℤ}
    (z₁ : Cochain F G n₁) (z₂ : Cochain G K (-n₂)) (z₃ : Cochain K L n₂) (h₁₂ : n₁ + (-n₂) = n₁₂) :
    (z₁.comp z₂ h₁₂).comp z₃
                             /-
                               C : Type u
                               inst✝³ : CategoryTheory.Category.{v, u} C
                               inst✝² : CategoryTheory.Preadditive C
                               R : Type u_1
                               inst✝¹ : Ring R
                               inst✝ : CategoryTheory.Linear R C
                               F G K L : CochainComplex C Int
                               n m n₁ n₂ n₁₂ : Int
                               z₁ : CochainComplex.HomComplex.Cochain F G n₁
                               z₂ : CochainComplex.HomComplex.Cochain G K (Neg.neg n₂)
                               z₃ : CochainComplex.HomComplex.Cochain K L n₂
                               h₁₂ : Eq (HAdd.hAdd n₁ (Neg.neg n₂)) n₁₂
                               ⊢ Eq (HAdd.hAdd n₁₂ n₂) n₁
                             -/
      (show n₁₂ + n₂ = n₁ by rw [← h₁₂, add_assoc, neg_add_cancel, add_zero]) =
                             /-
                               🎉 no goals
                             -/
      z₁.comp (z₂.comp z₃ (neg_add_cancel n₂)) (add_zero n₁) :=
                           /-
                             C : Type u
                             inst✝¹ : CategoryTheory.Category.{v, u} C
                             inst✝ : CategoryTheory.Preadditive C
                             F G K L : CochainComplex C Int
                             n₁ n₂ n₁₂ : Int
                             z₁ : CochainComplex.HomComplex.Cochain F G n₁
                             z₂ : CochainComplex.HomComplex.Cochain G K (Neg.neg n₂)
                             z₃ : CochainComplex.HomComplex.Cochain K L n₂
                             h₁₂ : Eq (HAdd.hAdd n₁ (Neg.neg n₂)) n₁₂
                             ⊢ Eq (HAdd.hAdd (HAdd.hAdd n₁ (Neg.neg n₂)) n₂) n₁
                           -/
  comp_assoc _ _ _ _ _ (by omega)
                           /-
                             🎉 no goals
                           -/


@[simp]
protected lemma zero_comp {n₁ n₂ n₁₂ : ℤ} (z₂ : Cochain G K n₂)
    (h : n₁ + n₂ = n₁₂) : (0 : Cochain F G n₁).comp z₂ h = 0 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    ⊢ Eq (CochainComplex.HomComplex.Cochain.comp 0 z₂ h) 0
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    p q : Int
    hpq : Eq (HAdd.hAdd p n₁₂) q
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.comp 0 z₂ h).v p q hpq) (CochainCompl …
  -/
  simp only [comp_v _ _ h p _ q rfl (by omega), zero_v, zero_comp]
  /-
    🎉 no goals
  -/


@[simp]
protected lemma add_comp {n₁ n₂ n₁₂ : ℤ} (z₁ z₁' : Cochain F G n₁) (z₂ : Cochain G K n₂)
    (h : n₁ + n₂ = n₁₂) : (z₁ + z₁').comp z₂ h = z₁.comp z₂ h + z₁'.comp z₂ h := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₁ z₁' : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    ⊢ Eq ((HAdd.hAdd z₁ z₁').comp z₂ h) (HAdd.hAdd (z₁.comp z₂ h) (z₁'.comp z₂ h))
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₁ z₁' : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    p q : Int
    hpq : Eq (HAdd.hAdd p n₁₂) q
    ⊢ Eq (((HAdd.hAdd z₁ z₁').comp z₂ h).v p q hpq) ((HAdd.hAdd (z₁.comp z₂ h) (z₁ …
  -/
  simp only [comp_v _ _ h p _ q rfl (by omega), add_v, add_comp]
  /-
    🎉 no goals
  -/


@[simp]
protected lemma sub_comp {n₁ n₂ n₁₂ : ℤ} (z₁ z₁' : Cochain F G n₁) (z₂ : Cochain G K n₂)
    (h : n₁ + n₂ = n₁₂) : (z₁ - z₁').comp z₂ h = z₁.comp z₂ h - z₁'.comp z₂ h := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₁ z₁' : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    ⊢ Eq ((HSub.hSub z₁ z₁').comp z₂ h) (HSub.hSub (z₁.comp z₂ h) (z₁'.comp z₂ h))
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₁ z₁' : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    p q : Int
    hpq : Eq (HAdd.hAdd p n₁₂) q
    ⊢ Eq (((HSub.hSub z₁ z₁').comp z₂ h).v p q hpq) ((HSub.hSub (z₁.comp z₂ h) (z₁ …
  -/
  simp only [comp_v _ _ h p _ q rfl (by omega), sub_v, sub_comp]
  /-
    🎉 no goals
  -/


@[simp]
protected lemma neg_comp {n₁ n₂ n₁₂ : ℤ} (z₁ : Cochain F G n₁) (z₂ : Cochain G K n₂)
    (h : n₁ + n₂ = n₁₂) : (-z₁).comp z₂ h = -z₁.comp z₂ h := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    ⊢ Eq ((Neg.neg z₁).comp z₂ h) (Neg.neg (z₁.comp z₂ h))
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    p q : Int
    hpq : Eq (HAdd.hAdd p n₁₂) q
    ⊢ Eq (((Neg.neg z₁).comp z₂ h).v p q hpq) ((Neg.neg (z₁.comp z₂ h)).v p q hpq)
  -/
  simp only [comp_v _ _ h p _ q rfl (by omega), neg_v, neg_comp]
  /-
    🎉 no goals
  -/


@[simp]
protected lemma smul_comp {n₁ n₂ n₁₂ : ℤ} (k : R) (z₁ : Cochain F G n₁) (z₂ : Cochain G K n₂)
    (h : n₁ + n₂ = n₁₂) : (k • z₁).comp z₂ h = k • (z₁.comp z₂ h) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    k : R
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    ⊢ Eq ((HSMul.hSMul k z₁).comp z₂ h) (HSMul.hSMul k (z₁.comp z₂ h))
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    k : R
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    p q : Int
    hpq : Eq (HAdd.hAdd p n₁₂) q
    ⊢ Eq (((HSMul.hSMul k z₁).comp z₂ h).v p q hpq) ((HSMul.hSMul k (z₁.comp z₂ h) …
  -/
  simp only [comp_v _ _ h p _ q rfl (by omega), smul_v, Linear.smul_comp]
  /-
    🎉 no goals
  -/


@[simp]
lemma units_smul_comp {n₁ n₂ n₁₂ : ℤ} (k : Rˣ) (z₁ : Cochain F G n₁) (z₂ : Cochain G K n₂)
    (h : n₁ + n₂ = n₁₂) : (k • z₁).comp z₂ h = k • (z₁.comp z₂ h) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    k : Units R
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    ⊢ Eq ((HSMul.hSMul k z₁).comp z₂ h) (HSMul.hSMul k (z₁.comp z₂ h))
  -/
  apply Cochain.smul_comp
  /-
    🎉 no goals
  -/


@[simp]
protected lemma id_comp {n : ℤ} (z₂ : Cochain F G n) :
    (Cochain.ofHom (𝟙 F)).comp z₂ (zero_add n) = z₂ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n : Int
    z₂ : CochainComplex.HomComplex.Cochain F G n
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.ofHom (CategoryTheory.CategoryStruct. …
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n : Int
    z₂ : CochainComplex.HomComplex.Cochain F G n
    p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    ⊢ Eq (((CochainComplex.HomComplex.Cochain.ofHom (CategoryTheory.CategoryStruct …
  -/
  simp only [zero_cochain_comp_v, ofHom_v, HomologicalComplex.id_f, id_comp]
  /-
    🎉 no goals
  -/


@[simp]
protected lemma comp_zero {n₁ n₂ n₁₂ : ℤ} (z₁ : Cochain F G n₁)
    (h : n₁ + n₂ = n₁₂) : z₁.comp (0 : Cochain G K n₂) h = 0 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    ⊢ Eq (z₁.comp 0 h) 0
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    p q : Int
    hpq : Eq (HAdd.hAdd p n₁₂) q
    ⊢ Eq ((z₁.comp 0 h).v p q hpq) (CochainComplex.HomComplex.Cochain.v 0 p q hpq)
  -/
  simp only [comp_v _ _ h p _ q rfl (by omega), zero_v, comp_zero]
  /-
    🎉 no goals
  -/


@[simp]
protected lemma comp_add {n₁ n₂ n₁₂ : ℤ} (z₁ : Cochain F G n₁) (z₂ z₂' : Cochain G K n₂)
    (h : n₁ + n₂ = n₁₂) : z₁.comp (z₂ + z₂') h = z₁.comp z₂ h + z₁.comp z₂' h := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ z₂' : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    ⊢ Eq (z₁.comp (HAdd.hAdd z₂ z₂') h) (HAdd.hAdd (z₁.comp z₂ h) (z₁.comp z₂' h))
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ z₂' : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    p q : Int
    hpq : Eq (HAdd.hAdd p n₁₂) q
    ⊢ Eq ((z₁.comp (HAdd.hAdd z₂ z₂') h).v p q hpq) ((HAdd.hAdd (z₁.comp z₂ h) (z₁ …
  -/
  simp only [comp_v _ _ h p _ q rfl (by omega), add_v, comp_add]
  /-
    🎉 no goals
  -/


@[simp]
protected lemma comp_sub {n₁ n₂ n₁₂ : ℤ} (z₁ : Cochain F G n₁) (z₂ z₂' : Cochain G K n₂)
    (h : n₁ + n₂ = n₁₂) : z₁.comp (z₂ - z₂') h = z₁.comp z₂ h - z₁.comp z₂' h := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ z₂' : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    ⊢ Eq (z₁.comp (HSub.hSub z₂ z₂') h) (HSub.hSub (z₁.comp z₂ h) (z₁.comp z₂' h))
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ z₂' : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    p q : Int
    hpq : Eq (HAdd.hAdd p n₁₂) q
    ⊢ Eq ((z₁.comp (HSub.hSub z₂ z₂') h).v p q hpq) ((HSub.hSub (z₁.comp z₂ h) (z₁ …
  -/
  simp only [comp_v _ _ h p _ q rfl (by omega), sub_v, comp_sub]
  /-
    🎉 no goals
  -/


@[simp]
protected lemma comp_neg {n₁ n₂ n₁₂ : ℤ} (z₁ : Cochain F G n₁) (z₂ : Cochain G K n₂)
    (h : n₁ + n₂ = n₁₂) : z₁.comp (-z₂) h = -z₁.comp z₂ h := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    ⊢ Eq (z₁.comp (Neg.neg z₂) h) (Neg.neg (z₁.comp z₂ h))
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    p q : Int
    hpq : Eq (HAdd.hAdd p n₁₂) q
    ⊢ Eq ((z₁.comp (Neg.neg z₂) h).v p q hpq) ((Neg.neg (z₁.comp z₂ h)).v p q hpq)
  -/
  simp only [comp_v _ _ h p _ q rfl (by omega), neg_v, comp_neg]
  /-
    🎉 no goals
  -/


@[simp]
protected lemma comp_smul {n₁ n₂ n₁₂ : ℤ} (z₁ : Cochain F G n₁) (k : R) (z₂ : Cochain G K n₂)
    (h : n₁ + n₂ = n₁₂ ) : z₁.comp (k • z₂) h = k • (z₁.comp z₂ h) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    k : R
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    ⊢ Eq (z₁.comp (HSMul.hSMul k z₂) h) (HSMul.hSMul k (z₁.comp z₂ h))
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    k : R
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    p q : Int
    hpq : Eq (HAdd.hAdd p n₁₂) q
    ⊢ Eq ((z₁.comp (HSMul.hSMul k z₂) h).v p q hpq) ((HSMul.hSMul k (z₁.comp z₂ h) …
  -/
  simp only [comp_v _ _ h p _ q rfl (by omega), smul_v, Linear.comp_smul]
  /-
    🎉 no goals
  -/


@[simp]
lemma comp_units_smul {n₁ n₂ n₁₂ : ℤ} (z₁ : Cochain F G n₁) (k : Rˣ) (z₂ : Cochain G K n₂)
    (h : n₁ + n₂ = n₁₂ ) : z₁.comp (k • z₂) h = k • (z₁.comp z₂ h) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    k : Units R
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    ⊢ Eq (z₁.comp (HSMul.hSMul k z₂) h) (HSMul.hSMul k (z₁.comp z₂ h))
  -/
  apply Cochain.comp_smul
  /-
    🎉 no goals
  -/


@[simp]
protected lemma comp_id {n : ℤ} (z₁ : Cochain F G n) :
    z₁.comp (Cochain.ofHom (𝟙 G)) (add_zero n) = z₁ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n
    ⊢ Eq (z₁.comp (CochainComplex.HomComplex.Cochain.ofHom (CategoryTheory.Categor …
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n
    p q : Int
    hpq : Eq (HAdd.hAdd p n) q
    ⊢ Eq ((z₁.comp (CochainComplex.HomComplex.Cochain.ofHom (CategoryTheory.Catego …
  -/
  simp only [comp_zero_cochain_v, ofHom_v, HomologicalComplex.id_f, comp_id]
  /-
    🎉 no goals
  -/


@[simp]
lemma ofHoms_comp (φ : ∀ (p : ℤ), F.X p ⟶ G.X p) (ψ : ∀ (p : ℤ), G.X p ⟶ K.X p) :
                                                                                /-
                                                                                  C : Type u
                                                                                  inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                                  inst✝ : CategoryTheory.Preadditive C
                                                                                  F G K : CochainComplex C Int
                                                                                  φ : (p : Int) → Quiver.Hom (F.X p) (G.X p)
                                                                                  ψ : (p : Int) → Quiver.Hom (G.X p) (K.X p)
                                                                                  ⊢ Eq ((CochainComplex.HomComplex.Cochain.ofHoms φ).comp (CochainComplex.HomCom …
                                                                                -/
    (ofHoms φ).comp (ofHoms ψ) (zero_add 0) = ofHoms (fun p => φ p ≫ ψ p) := by aesop_cat
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


@[simp]
lemma ofHom_comp (f : F ⟶ G) (g : G ⟶ K) :
    ofHom (f ≫ g) = (ofHom f).comp (ofHom g) (zero_add 0) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    f : Quiver.Hom F G
    g : Quiver.Hom G K
    ⊢ Eq (CochainComplex.HomComplex.Cochain.ofHom (CategoryTheory.CategoryStruct.c …
  -/
  simp only [ofHom, HomologicalComplex.comp_f, ofHoms_comp]
  /-
    🎉 no goals
  -/


/-- The differential on a cochain complex, as a cochain of degree `1`. -/
def diff : Cochain K K 1 := Cochain.mk (fun p q _ => K.d p q)


@[simp]
lemma diff_v (p q : ℤ) (hpq : p + 1 = q) : (diff K).v p q hpq = K.d p q := rfl


/-- The differential on the complex of morphisms between cochain complexes. -/
def δ (z : Cochain F G n) : Cochain F G m :=
  Cochain.mk (fun p q hpq => z.v p (p + n) rfl ≫ G.d (p + n) q +
                                                            /-
                                                              C : Type u
                                                              inst✝³ : CategoryTheory.Category.{v, u} C
                                                              inst✝² : CategoryTheory.Preadditive C
                                                              R : Type u_1
                                                              inst✝¹ : Ring R
                                                              inst✝ : CategoryTheory.Linear R C
                                                              F G K L : CochainComplex C Int
                                                              n m : Int
                                                              z : CochainComplex.HomComplex.Cochain F G n
                                                              p q : Int
                                                              hpq : Eq (HAdd.hAdd p m) q
                                                              ⊢ Eq (HAdd.hAdd (HSub.hSub (HAdd.hAdd p m) n) n) q
                                                            -/
    m.negOnePow • F.d p (p + m - n) ≫ z.v (p + m - n) q (by rw [hpq, sub_add_cancel]))
                                                            /-
                                                              🎉 no goals
                                                            -/


lemma δ_v (hnm : n + 1 = m) (z : Cochain F G n) (p q : ℤ) (hpq : p + m = q) (q₁ q₂ : ℤ)
    (hq₁ : q₁ = q - 1) (hq₂ : p + 1 = q₂) : (δ n m z).v p q hpq =
                 /-
                   C : Type u
                   inst✝³ : CategoryTheory.Category.{v, u} C
                   inst✝² : CategoryTheory.Preadditive C
                   R : Type u_1
                   inst✝¹ : Ring R
                   inst✝ : CategoryTheory.Linear R C
                   F G K L : CochainComplex C Int
                   n m : Int
                   hnm : Eq (HAdd.hAdd n 1) m
                   z : CochainComplex.HomComplex.Cochain F G n
                   p q : Int
                   hpq : Eq (HAdd.hAdd p m) q
                   q₁ q₂ : Int
                   hq₁ : Eq q₁ (HSub.hSub q 1)
                   hq₂ : Eq (HAdd.hAdd p 1) q₂
                   ⊢ Eq (HAdd.hAdd p n) q₁
                 -/
    z.v p q₁ (by rw [hq₁, ← hpq, ← hnm, ← add_assoc, add_sub_cancel_right]) ≫ G.d q₁ q
                 /-
                   🎉 no goals
                 -/
      + m.negOnePow • F.d p q₂ ≫ z.v q₂ q
              /-
                C : Type u
                inst✝³ : CategoryTheory.Category.{v, u} C
                inst✝² : CategoryTheory.Preadditive C
                R : Type u_1
                inst✝¹ : Ring R
                inst✝ : CategoryTheory.Linear R C
                F G K L : CochainComplex C Int
                n m : Int
                hnm : Eq (HAdd.hAdd n 1) m
                z : CochainComplex.HomComplex.Cochain F G n
                p q : Int
                hpq : Eq (HAdd.hAdd p m) q
                q₁ q₂ : Int
                hq₁ : Eq q₁ (HSub.hSub q 1)
                hq₂ : Eq (HAdd.hAdd p 1) q₂
                ⊢ Eq (HAdd.hAdd q₂ n) q
              -/
          (by rw [← hq₂, add_assoc, add_comm 1, hnm, hpq]) := by
              /-
                🎉 no goals
              -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n m : Int
    hnm : Eq (HAdd.hAdd n 1) m
    z : CochainComplex.HomComplex.Cochain F G n
    p q : Int
    hpq : Eq (HAdd.hAdd p m) q
    q₁ q₂ : Int
    hq₁ : Eq q₁ (HSub.hSub q 1)
    hq₂ : Eq (HAdd.hAdd p 1) q₂
    ⊢ Eq ((CochainComplex.HomComplex.δ n m z).v p q hpq) (HAdd.hAdd (CategoryTheor …
  -/
  obtain rfl : q₁ = p + n := by omega
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n m : Int
    hnm : Eq (HAdd.hAdd n 1) m
    z : CochainComplex.HomComplex.Cochain F G n
    p q : Int
    hpq : Eq (HAdd.hAdd p m) q
    q₂ : Int
    hq₂ : Eq (HAdd.hAdd p 1) q₂
    hq₁ : Eq (HAdd.hAdd p n) (HSub.hSub q 1)
    ⊢ Eq ((CochainComplex.HomComplex.δ n m z).v p q hpq) (HAdd.hAdd (CategoryTheor …
  -/
  obtain rfl : q₂ = p + m - n := by omega
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n m : Int
    hnm : Eq (HAdd.hAdd n 1) m
    z : CochainComplex.HomComplex.Cochain F G n
    p q : Int
    hpq : Eq (HAdd.hAdd p m) q
    hq₁ : Eq (HAdd.hAdd p n) (HSub.hSub q 1)
    hq₂ : Eq (HAdd.hAdd p 1) (HSub.hSub (HAdd.hAdd p m) n)
    ⊢ Eq ((CochainComplex.HomComplex.δ n m z).v p q hpq) (HAdd.hAdd (CategoryTheor …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma δ_shape (hnm : ¬ n + 1 = m) (z : Cochain F G n) : δ n m z = 0 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n m : Int
    hnm : Not (Eq (HAdd.hAdd n 1) m)
    z : CochainComplex.HomComplex.Cochain F G n
    ⊢ Eq (CochainComplex.HomComplex.δ n m z) 0
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n m : Int
    hnm : Not (Eq (HAdd.hAdd n 1) m)
    z : CochainComplex.HomComplex.Cochain F G n
    p q : Int
    hpq : Eq (HAdd.hAdd p m) q
    ⊢ Eq ((CochainComplex.HomComplex.δ n m z).v p q hpq) (CochainComplex.HomComple …
  -/
  dsimp only [δ]
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n m : Int
    hnm : Not (Eq (HAdd.hAdd n 1) m)
    z : CochainComplex.HomComplex.Cochain F G n
    p q : Int
    hpq : Eq (HAdd.hAdd p m) q
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.mk fun p q hpq => HAdd.hAdd (Category …
  -/
  rw [Cochain.mk_v, Cochain.zero_v, F.shape, G.shape, comp_zero, zero_add, zero_comp, smul_zero]
  all_goals
    simp only [ComplexShape.up_Rel]
    exact fun _ => hnm (by omega)


/-- The differential on the complex of morphisms between cochain complexes, as a linear map. -/
@[simps!]
def δ_hom : Cochain F G n →ₗ[R] Cochain F G m where
  toFun := δ n m
  map_add' α β := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CategoryTheory.Linear R C
      F G K L : CochainComplex C Int
      n m : Int
      α β : CochainComplex.HomComplex.Cochain F G n
      ⊢ Eq (CochainComplex.HomComplex.δ n m (HAdd.hAdd α β)) (HAdd.hAdd (CochainComp …
    -/
    by_cases h : n + 1 = m
      /-
        case pos
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Preadditive C
        R : Type u_1
        inst✝¹ : Ring R
        inst✝ : CategoryTheory.Linear R C
        F G K L : CochainComplex C Int
        n m : Int
        α β : CochainComplex.HomComplex.Cochain F G n
        h : Eq (HAdd.hAdd n 1) m
        ⊢ Eq (CochainComplex.HomComplex.δ n m (HAdd.hAdd α β)) (HAdd.hAdd (CochainComp …
      -/
    · ext p q hpq
      /-
        case pos.h
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Preadditive C
        R : Type u_1
        inst✝¹ : Ring R
        inst✝ : CategoryTheory.Linear R C
        F G K L : CochainComplex C Int
        n m : Int
        α β : CochainComplex.HomComplex.Cochain F G n
        h : Eq (HAdd.hAdd n 1) m
        p q : Int
        hpq : Eq (HAdd.hAdd p m) q
        ⊢ Eq ((CochainComplex.HomComplex.δ n m (HAdd.hAdd α β)).v p q hpq) ((HAdd.hAdd …
      -/
      dsimp
      /-
        case pos.h
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Preadditive C
        R : Type u_1
        inst✝¹ : Ring R
        inst✝ : CategoryTheory.Linear R C
        F G K L : CochainComplex C Int
        n m : Int
        α β : CochainComplex.HomComplex.Cochain F G n
        h : Eq (HAdd.hAdd n 1) m
        p q : Int
        hpq : Eq (HAdd.hAdd p m) q
        ⊢ Eq ((CochainComplex.HomComplex.δ n m (HAdd.hAdd α β)).v p q hpq) (HAdd.hAdd  …
      -/
      simp only [δ_v n m h _ p q hpq _ _ rfl rfl, Cochain.add_v, add_comp, comp_add, smul_add]
      /-
        case pos.h
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Preadditive C
        R : Type u_1
        inst✝¹ : Ring R
        inst✝ : CategoryTheory.Linear R C
        F G K L : CochainComplex C Int
        n m : Int
        α β : CochainComplex.HomComplex.Cochain F G n
        h : Eq (HAdd.hAdd n 1) m
        p q : Int
        hpq : Eq (HAdd.hAdd p m) q
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (α.v p (HSub.hS …
      -/
      /-
        🎉 no goals
      -/
      abel
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Preadditive C
        R : Type u_1
        inst✝¹ : Ring R
        inst✝ : CategoryTheory.Linear R C
        F G K L : CochainComplex C Int
        n m : Int
        α β : CochainComplex.HomComplex.Cochain F G n
        h : Not (Eq (HAdd.hAdd n 1) m)
        ⊢ Eq (CochainComplex.HomComplex.δ n m (HAdd.hAdd α β)) (HAdd.hAdd (CochainComp …
      -/
    · simp only [δ_shape _ _ h, add_zero]
      /-
        🎉 no goals
      -/
  map_smul' r a := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CategoryTheory.Linear R C
      F G K L : CochainComplex C Int
      n m : Int
      r : R
      a : CochainComplex.HomComplex.Cochain F G n
      ⊢ Eq ({ toFun := CochainComplex.HomComplex.δ n m, map_add' := ⋯ }.toFun (HSMul …
    -/
    by_cases h : n + 1 = m
      /-
        case pos
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Preadditive C
        R : Type u_1
        inst✝¹ : Ring R
        inst✝ : CategoryTheory.Linear R C
        F G K L : CochainComplex C Int
        n m : Int
        r : R
        a : CochainComplex.HomComplex.Cochain F G n
        h : Eq (HAdd.hAdd n 1) m
        ⊢ Eq ({ toFun := CochainComplex.HomComplex.δ n m, map_add' := ⋯ }.toFun (HSMul …
      -/
    · ext p q hpq
      /-
        case pos.h
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Preadditive C
        R : Type u_1
        inst✝¹ : Ring R
        inst✝ : CategoryTheory.Linear R C
        F G K L : CochainComplex C Int
        n m : Int
        r : R
        a : CochainComplex.HomComplex.Cochain F G n
        h : Eq (HAdd.hAdd n 1) m
        p q : Int
        hpq : Eq (HAdd.hAdd p m) q
        ⊢ Eq (({ toFun := CochainComplex.HomComplex.δ n m, map_add' := ⋯ }.toFun (HSMu …
      -/
      dsimp
      simp only [δ_v n m h _ p q hpq _ _ rfl rfl, Cochain.smul_v, Linear.comp_smul,
        Linear.smul_comp, smul_add, add_right_inj, smul_comm m.negOnePow r]
      /-
        case neg
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Preadditive C
        R : Type u_1
        inst✝¹ : Ring R
        inst✝ : CategoryTheory.Linear R C
        F G K L : CochainComplex C Int
        n m : Int
        r : R
        a : CochainComplex.HomComplex.Cochain F G n
        h : Not (Eq (HAdd.hAdd n 1) m)
        ⊢ Eq ({ toFun := CochainComplex.HomComplex.δ n m, map_add' := ⋯ }.toFun (HSMul …
      -/
    · simp only [δ_shape _ _ h, smul_zero]
      /-
        🎉 no goals
      -/


@[simp] lemma δ_add (z₁ z₂ : Cochain F G n) : δ n m (z₁ + z₂) = δ n m z₁ + δ n m z₂ :=
  (δ_hom ℤ F G n m).map_add z₁ z₂


@[simp] lemma δ_sub (z₁ z₂ : Cochain F G n) : δ n m (z₁ - z₂) = δ n m z₁ - δ n m z₂ :=
  (δ_hom ℤ F G n m).map_sub z₁ z₂


@[simp] lemma δ_zero : δ n m (0 : Cochain F G n) = 0 := (δ_hom ℤ F G n m).map_zero


@[simp] lemma δ_neg (z : Cochain F G n) : δ n m (-z) = - δ n m z :=
  (δ_hom ℤ F G n m).map_neg z


@[simp] lemma δ_smul (k : R) (z : Cochain F G n) : δ n m (k • z) = k • δ n m z :=
  (δ_hom R F G n m).map_smul k z


@[simp] lemma δ_units_smul (k : Rˣ) (z : Cochain F G n) : δ n m (k • z) = k • δ n m z :=
  δ_smul ..


lemma δ_δ (n₀ n₁ n₂ : ℤ) (z : Cochain F G n₀) : δ n₁ n₂ (δ n₀ n₁ z) = 0 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n₀ n₁ n₂ : Int
    z : CochainComplex.HomComplex.Cochain F G n₀
    ⊢ Eq (CochainComplex.HomComplex.δ n₁ n₂ (CochainComplex.HomComplex.δ n₀ n₁ z)) 0
  -/
  by_cases h₁₂ : n₁ + 1 = n₂; swap
    /-
      case neg
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      n₀ n₁ n₂ : Int
      z : CochainComplex.HomComplex.Cochain F G n₀
      h₁₂ : Not (Eq (HAdd.hAdd n₁ 1) n₂)
      ⊢ Eq (CochainComplex.HomComplex.δ n₁ n₂ (CochainComplex.HomComplex.δ n₀ n₁ z)) 0
    -/
  · rw [δ_shape _ _ h₁₂]
    /-
      🎉 no goals
    -/
  /-
    case pos
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n₀ n₁ n₂ : Int
    z : CochainComplex.HomComplex.Cochain F G n₀
    h₁₂ : Eq (HAdd.hAdd n₁ 1) n₂
    ⊢ Eq (CochainComplex.HomComplex.δ n₁ n₂ (CochainComplex.HomComplex.δ n₀ n₁ z)) 0
  -/
  by_cases h₀₁ : n₀ + 1 = n₁; swap
    /-
      case neg
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      n₀ n₁ n₂ : Int
      z : CochainComplex.HomComplex.Cochain F G n₀
      h₁₂ : Eq (HAdd.hAdd n₁ 1) n₂
      h₀₁ : Not (Eq (HAdd.hAdd n₀ 1) n₁)
      ⊢ Eq (CochainComplex.HomComplex.δ n₁ n₂ (CochainComplex.HomComplex.δ n₀ n₁ z)) 0
    -/
  · rw [δ_shape _ _ h₀₁, δ_zero]
    /-
      🎉 no goals
    -/
  /-
    case pos
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n₀ n₁ n₂ : Int
    z : CochainComplex.HomComplex.Cochain F G n₀
    h₁₂ : Eq (HAdd.hAdd n₁ 1) n₂
    h₀₁ : Eq (HAdd.hAdd n₀ 1) n₁
    ⊢ Eq (CochainComplex.HomComplex.δ n₁ n₂ (CochainComplex.HomComplex.δ n₀ n₁ z)) 0
  -/
  ext p q hpq
  /-
    case pos.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n₀ n₁ n₂ : Int
    z : CochainComplex.HomComplex.Cochain F G n₀
    h₁₂ : Eq (HAdd.hAdd n₁ 1) n₂
    h₀₁ : Eq (HAdd.hAdd n₀ 1) n₁
    p q : Int
    hpq : Eq (HAdd.hAdd p n₂) q
    ⊢ Eq ((CochainComplex.HomComplex.δ n₁ n₂ (CochainComplex.HomComplex.δ n₀ n₁ z) …
  -/
  dsimp
  simp only [δ_v n₁ n₂ h₁₂ _ p q hpq _ _ rfl rfl,
    δ_v n₀ n₁ h₀₁ z p (q-1) (by omega) (q-2) _ (by omega) rfl,
    δ_v n₀ n₁ h₀₁ z (p+1) q (by omega) _ (p+2) rfl (by omega),
    ← h₁₂, Int.negOnePow_succ, add_comp, assoc,
    HomologicalComplex.d_comp_d, comp_zero, zero_add, comp_add,
    HomologicalComplex.d_comp_d_assoc, zero_comp, smul_zero,
    add_zero, add_neg_cancel, Units.neg_smul,
    Linear.units_smul_comp, Linear.comp_units_smul]


lemma δ_comp {n₁ n₂ n₁₂ : ℤ} (z₁ : Cochain F G n₁) (z₂ : Cochain G K n₂) (h : n₁ + n₂ = n₁₂)
    (m₁ m₂ m₁₂ : ℤ) (h₁₂ : n₁₂ + 1 = m₁₂) (h₁ : n₁ + 1 = m₁) (h₂ : n₂ + 1 = m₂) :
                                                        /-
                                                          C : Type u
                                                          inst✝³ : CategoryTheory.Category.{v, u} C
                                                          inst✝² : CategoryTheory.Preadditive C
                                                          R : Type u_1
                                                          inst✝¹ : Ring R
                                                          inst✝ : CategoryTheory.Linear R C
                                                          F G K L : CochainComplex C Int
                                                          n m n₁ n₂ n₁₂ : Int
                                                          z₁ : CochainComplex.HomComplex.Cochain F G n₁
                                                          z₂ : CochainComplex.HomComplex.Cochain G K n₂
                                                          h : Eq (HAdd.hAdd n₁ n₂) n₁₂
                                                          m₁ m₂ m₁₂ : Int
                                                          h₁₂ : Eq (HAdd.hAdd n₁₂ 1) m₁₂
                                                          h₁ : Eq (HAdd.hAdd n₁ 1) m₁
                                                          h₂ : Eq (HAdd.hAdd n₂ 1) m₂
                                                          ⊢ Eq (HAdd.hAdd n₁ m₂) m₁₂
                                                        -/
    δ n₁₂ m₁₂ (z₁.comp z₂ h) = z₁.comp (δ n₂ m₂ z₂) (by rw [← h₁₂, ← h₂, ← h, add_assoc]) +
                                                        /-
                                                          🎉 no goals
                                                        -/
      n₂.negOnePow • (δ n₁ m₁ z₁).comp z₂
            /-
              C : Type u
              inst✝³ : CategoryTheory.Category.{v, u} C
              inst✝² : CategoryTheory.Preadditive C
              R : Type u_1
              inst✝¹ : Ring R
              inst✝ : CategoryTheory.Linear R C
              F G K L : CochainComplex C Int
              n m n₁ n₂ n₁₂ : Int
              z₁ : CochainComplex.HomComplex.Cochain F G n₁
              z₂ : CochainComplex.HomComplex.Cochain G K n₂
              h : Eq (HAdd.hAdd n₁ n₂) n₁₂
              m₁ m₂ m₁₂ : Int
              h₁₂ : Eq (HAdd.hAdd n₁₂ 1) m₁₂
              h₁ : Eq (HAdd.hAdd n₁ 1) m₁
              h₂ : Eq (HAdd.hAdd n₂ 1) m₂
              ⊢ Eq (HAdd.hAdd m₁ n₂) m₁₂
            -/
        (by rw [← h₁₂, ← h₁, ← h, add_assoc, add_comm 1, add_assoc]) := by
            /-
              🎉 no goals
            -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ n₁₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    m₁ m₂ m₁₂ : Int
    h₁₂ : Eq (HAdd.hAdd n₁₂ 1) m₁₂
    h₁ : Eq (HAdd.hAdd n₁ 1) m₁
    h₂ : Eq (HAdd.hAdd n₂ 1) m₂
    ⊢ Eq (CochainComplex.HomComplex.δ n₁₂ m₁₂ (z₁.comp z₂ h)) (HAdd.hAdd (z₁.comp  …
  -/
  subst h₁₂ h₁ h₂ h
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    ⊢ Eq (CochainComplex.HomComplex.δ (HAdd.hAdd n₁ n₂) (HAdd.hAdd (HAdd.hAdd n₁ n …
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    p q : Int
    hpq : Eq (HAdd.hAdd p (HAdd.hAdd (HAdd.hAdd n₁ n₂) 1)) q
    ⊢ Eq ((CochainComplex.HomComplex.δ (HAdd.hAdd n₁ n₂) (HAdd.hAdd (HAdd.hAdd n₁  …
  -/
  dsimp
  rw [z₁.comp_v _ (add_assoc n₁ n₂ 1).symm p _ q rfl (by omega),
    Cochain.comp_v _ _ (show n₁ + 1 + n₂ = n₁ + n₂ + 1 by omega) p (p+n₁+1) q
      (by omega) (by omega),
    δ_v (n₁ + n₂) _ rfl (z₁.comp z₂ rfl) p q hpq (p + n₁ + n₂) _ (by omega) rfl,
    z₁.comp_v z₂ rfl p _ _ rfl rfl,
    z₁.comp_v z₂ rfl (p+1) (p+n₁+1) q (by omega) (by omega),
    δ_v n₂ (n₂+1) rfl z₂ (p+n₁) q (by omega) (p+n₁+n₂) _ (by omega) rfl,
    δ_v n₁ (n₁+1) rfl z₁ p (p+n₁+1) (by omega) (p+n₁) _ (by omega) rfl]
  simp only [assoc, comp_add, add_comp, Int.negOnePow_succ, Int.negOnePow_add n₁ n₂,
    Units.neg_smul, comp_neg, neg_comp, smul_neg, smul_smul, Linear.units_smul_comp,
    mul_comm n₁.negOnePow n₂.negOnePow, Linear.comp_units_smul, smul_add]
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n₁ n₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    p q : Int
    hpq : Eq (HAdd.hAdd p (HAdd.hAdd (HAdd.hAdd n₁ n₂) 1)) q
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (z₁.v p (HAdd.hAdd p n₁) ⋯ …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


lemma δ_zero_cochain_comp {n₂ : ℤ} (z₁ : Cochain F G 0) (z₂ : Cochain G K n₂)
    (m₂ : ℤ) (h₂ : n₂ + 1 = m₂) :
    δ n₂ m₂ (z₁.comp z₂ (zero_add n₂)) =
      z₁.comp (δ n₂ m₂ z₂) (zero_add m₂) +
                                             /-
                                               C : Type u
                                               inst✝³ : CategoryTheory.Category.{v, u} C
                                               inst✝² : CategoryTheory.Preadditive C
                                               R : Type u_1
                                               inst✝¹ : Ring R
                                               inst✝ : CategoryTheory.Linear R C
                                               F G K L : CochainComplex C Int
                                               n m n₂ : Int
                                               z₁ : CochainComplex.HomComplex.Cochain F G 0
                                               z₂ : CochainComplex.HomComplex.Cochain G K n₂
                                               m₂ : Int
                                               h₂ : Eq (HAdd.hAdd n₂ 1) m₂
                                               ⊢ Eq (HAdd.hAdd 1 n₂) m₂
                                             -/
      n₂.negOnePow • ((δ 0 1 z₁).comp z₂ (by rw [add_comm, h₂])) :=
                                             /-
                                               🎉 no goals
                                             -/
  δ_comp z₁ z₂ (zero_add n₂) 1 m₂ m₂ h₂ (zero_add 1) h₂


lemma δ_comp_zero_cochain {n₁ : ℤ} (z₁ : Cochain F G n₁) (z₂ : Cochain G K 0)
    (m₁ : ℤ) (h₁ : n₁ + 1 = m₁) :
    δ n₁ m₁ (z₁.comp z₂ (add_zero n₁)) =
      z₁.comp (δ 0 1 z₂) h₁ + (δ n₁ m₁ z₁).comp z₂ (add_zero m₁) := by
  simp only [δ_comp z₁ z₂ (add_zero n₁) m₁ 1 m₁ h₁ h₁ (zero_add 1), one_smul,
    Int.negOnePow_zero]


@[simp]
lemma δ_zero_cochain_v (z : Cochain F G 0) (p q : ℤ) (hpq : p + 1 = q) :
    (δ 0 1 z).v p q hpq = z.v p p (add_zero p) ≫ G.d p q - F.d p q ≫ z.v q q (add_zero q) := by
  simp only [δ_v 0 1 (zero_add 1) z p q hpq p q (by omega) hpq, zero_add,
    Int.negOnePow_one, Units.neg_smul, one_smul, sub_eq_add_neg]


@[simp]
lemma δ_ofHom {p : ℤ} (φ : F ⟶ G) : δ 0 p (Cochain.ofHom φ) = 0 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    p : Int
    φ : Quiver.Hom F G
    ⊢ Eq (CochainComplex.HomComplex.δ 0 p (CochainComplex.HomComplex.Cochain.ofHom …
  -/
  by_cases h : p = 1
    /-
      case pos
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      p : Int
      φ : Quiver.Hom F G
      h : Eq p 1
      ⊢ Eq (CochainComplex.HomComplex.δ 0 p (CochainComplex.HomComplex.Cochain.ofHom …
    -/
  · subst h
    /-
      case pos
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      ⊢ Eq (CochainComplex.HomComplex.δ 0 1 (CochainComplex.HomComplex.Cochain.ofHom …
    -/
    ext
    /-
      case pos.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      p✝ q✝ : Int
      hpq✝ : Eq (HAdd.hAdd p✝ 1) q✝
      ⊢ Eq ((CochainComplex.HomComplex.δ 0 1 (CochainComplex.HomComplex.Cochain.ofHo …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      p : Int
      φ : Quiver.Hom F G
      h : Not (Eq p 1)
      ⊢ Eq (CochainComplex.HomComplex.δ 0 p (CochainComplex.HomComplex.Cochain.ofHom …
    -/
  · rw [δ_shape]
    /-
      case neg.hnm
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      p : Int
      φ : Quiver.Hom F G
      h : Not (Eq p 1)
      ⊢ Not (Eq (HAdd.hAdd 0 1) p)
    -/
    omega
    /-
      🎉 no goals
    -/


@[simp]
lemma δ_ofHomotopy {φ₁ φ₂ : F ⟶ G} (h : Homotopy φ₁ φ₂) :
    δ (-1) 0 (Cochain.ofHomotopy h) = Cochain.ofHom φ₁ - Cochain.ofHom φ₂ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ₁ φ₂ : Quiver.Hom F G
    h : Homotopy φ₁ φ₂
    ⊢ Eq (CochainComplex.HomComplex.δ (-1) 0 (CochainComplex.HomComplex.Cochain.of …
  -/
  ext p
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ₁ φ₂ : Quiver.Hom F G
    h : Homotopy φ₁ φ₂
    p : Int
    ⊢ Eq ((CochainComplex.HomComplex.δ (-1) 0 (CochainComplex.HomComplex.Cochain.o …
  -/
  have eq := h.comm p
  rw [dNext_eq h.hom (show (ComplexShape.up ℤ).Rel p (p+1) by simp),
    prevD_eq h.hom (show (ComplexShape.up ℤ).Rel (p-1) p by simp)] at eq
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ₁ φ₂ : Quiver.Hom F G
    h : Homotopy φ₁ φ₂
    p : Int
    eq : Eq (φ₁.f p) (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (F. …
    ⊢ Eq ((CochainComplex.HomComplex.δ (-1) 0 (CochainComplex.HomComplex.Cochain.o …
  -/
  rw [Cochain.ofHomotopy, δ_v (-1) 0 (neg_add_cancel 1) _ p p (add_zero p) (p-1) (p+1) rfl rfl]
  simp only [Cochain.mk_v, neg_add_cancel, one_smul, Int.negOnePow_zero,
    Cochain.sub_v, Cochain.ofHom_v, eq]
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ₁ φ₂ : Quiver.Hom F G
    h : Homotopy φ₁ φ₂
    p : Int
    eq : Eq (φ₁.f p) (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (F. …
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (h.hom p (HSub.hSub p 1))  …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


lemma δ_neg_one_cochain (z : Cochain F G (-1)) :
    δ (-1) 0 z = Cochain.ofHom (Homotopy.nullHomotopicMap'
                                  /-
                                    C : Type u
                                    inst✝³ : CategoryTheory.Category.{v, u} C
                                    inst✝² : CategoryTheory.Preadditive C
                                    R : Type u_1
                                    inst✝¹ : Ring R
                                    inst✝ : CategoryTheory.Linear R C
                                    F G K L : CochainComplex C Int
                                    n m : Int
                                    z : CochainComplex.HomComplex.Cochain F G (-1)
                                    i j : Int
                                    hij : (ComplexShape.up Int).Rel j i
                                    ⊢ Eq (HAdd.hAdd i (-1)) j
                                  -/
      (fun i j hij => z.v i j (by dsimp at hij; rw [← hij, add_neg_cancel_right]))) := by
                                                /-
                                                  🎉 no goals
                                                -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    z : CochainComplex.HomComplex.Cochain F G (-1)
    ⊢ Eq (CochainComplex.HomComplex.δ (-1) 0 z) (CochainComplex.HomComplex.Cochain …
  -/
  ext p
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    z : CochainComplex.HomComplex.Cochain F G (-1)
    p : Int
    ⊢ Eq ((CochainComplex.HomComplex.δ (-1) 0 z).v p p ⋯) ((CochainComplex.HomComp …
  -/
  rw [δ_v (-1) 0 (neg_add_cancel 1) _ p p (add_zero p) (p-1) (p+1) rfl rfl]
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    z : CochainComplex.HomComplex.Cochain F G (-1)
    p : Int
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (z.v p (HSub.hSub p 1) ⋯)  …
  -/
  simp only [neg_add_cancel, one_smul, Cochain.ofHom_v, Int.negOnePow_zero]
  rw [Homotopy.nullHomotopicMap'_f (show (ComplexShape.up ℤ).Rel (p-1) p by simp)
    (show (ComplexShape.up ℤ).Rel p (p+1) by simp)]
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    z : CochainComplex.HomComplex.Cochain F G (-1)
    p : Int
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (z.v p (HSub.hSub p 1) ⋯)  …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


/-- The cochain complex of homomorphisms between two cochain complexes `F` and `G`.
In degree `n : ℤ`, it consists of the abelian group `HomComplex.Cochain F G n`. -/
-- We also constructed the `d_apply` lemma using `@[simps]`
-- until we made `AddCommGrp.coe_of` a simp lemma,
-- after which the simp normal form linter complains.
-- It was not used a simp lemma in Mathlib.
-- Possible solution: higher priority function coercions that remove the `of`?
-- @[simp]
@[simps! X]
def HomComplex : CochainComplex AddCommGrp ℤ where
  X i := AddCommGrp.of (Cochain F G i)
  d i j := AddCommGrp.ofHom (δ_hom ℤ F G i j)
                      /-
                        C : Type u
                        inst✝³ : CategoryTheory.Category.{v, u} C
                        inst✝² : CategoryTheory.Preadditive C
                        R : Type u_1
                        inst✝¹ : Ring R
                        inst✝ : CategoryTheory.Linear R C
                        F G K L : CochainComplex C Int
                        n m x✝¹ x✝ : Int
                        hij : Not ((ComplexShape.up Int).Rel x✝¹ x✝)
                        ⊢ Eq ((fun i j => AddCommGrp.ofHom ↑(CochainComplex.HomComplex.δ_hom Int F G i …
                      -/
  shape _ _ hij := by ext; apply δ_shape _ _ hij
                           /-
                             🎉 no goals
                           -/
                             /-
                               C : Type u
                               inst✝³ : CategoryTheory.Category.{v, u} C
                               inst✝² : CategoryTheory.Preadditive C
                               R : Type u_1
                               inst✝¹ : Ring R
                               inst✝ : CategoryTheory.Linear R C
                               F G K L : CochainComplex C Int
                               n m x✝⁴ x✝³ x✝² : Int
                               x✝¹ : (ComplexShape.up Int).Rel x✝⁴ x✝³
                               x✝ : (ComplexShape.up Int).Rel x✝³ x✝²
                               ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i j => AddCommGrp.ofHom ↑(Cocha …
                             -/
  d_comp_d' _ _ _ _ _  := by ext; apply δ_δ
                                  /-
                                    🎉 no goals
                                  -/


/-- The subgroup of cocycles in `Cochain F G n`. -/
def cocycle : AddSubgroup (Cochain F G n) :=
  AddMonoidHom.ker (δ_hom ℤ F G n (n + 1)).toAddMonoidHom


/-- The type of `n`-cocycles, as a subtype of `Cochain F G n`. -/
def Cocycle : Type v := cocycle F G n


instance : AddCommGroup (Cocycle F G n) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    F G K L : CochainComplex C Int
    n m : Int
    ⊢ AddCommGroup (CochainComplex.HomComplex.Cocycle F G n)
  -/
  dsimp only [Cocycle]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CategoryTheory.Linear R C
    F G K L : CochainComplex C Int
    n m : Int
    ⊢ AddCommGroup (Subtype fun x => Membership.mem (CochainComplex.HomComplex.coc …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma mem_iff (hnm : n + 1 = m) (z : Cochain F G n) :
                                          /-
                                            C : Type u
                                            inst✝¹ : CategoryTheory.Category.{v, u} C
                                            inst✝ : CategoryTheory.Preadditive C
                                            F G : CochainComplex C Int
                                            n m : Int
                                            hnm : Eq (HAdd.hAdd n 1) m
                                            z : CochainComplex.HomComplex.Cochain F G n
                                            ⊢ Iff (Membership.mem (CochainComplex.HomComplex.cocycle F G n) z) (Eq (Cochai …
                                          -/
    z ∈ cocycle F G n ↔ δ n m z = 0 := by subst hnm; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/


instance : Coe (Cocycle F G n) (Cochain F G n) where
  coe x := x.1


@[ext]
lemma ext (z₁ z₂ : Cocycle F G n) (h : (z₁ : Cochain F G n) = z₂) : z₁ = z₂ :=
  Subtype.ext h


instance : SMul R (Cocycle F G n) where
  smul r z := ⟨r • z.1, by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CategoryTheory.Linear R C
      F G K L : CochainComplex C Int
      n m : Int
      r : R
      z : CochainComplex.HomComplex.Cocycle F G n
      ⊢ Membership.mem (CochainComplex.HomComplex.cocycle F G n) (HSMul.hSMul r ↑z)
    -/
    have hz := z.2
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CategoryTheory.Linear R C
      F G K L : CochainComplex C Int
      n m : Int
      r : R
      z : CochainComplex.HomComplex.Cocycle F G n
      hz : Membership.mem (CochainComplex.HomComplex.cocycle F G n) ↑z
      ⊢ Membership.mem (CochainComplex.HomComplex.cocycle F G n) (HSMul.hSMul r ↑z)
    -/
    rw [mem_iff n (n + 1) rfl] at hz ⊢
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CategoryTheory.Linear R C
      F G K L : CochainComplex C Int
      n m : Int
      r : R
      z : CochainComplex.HomComplex.Cocycle F G n
      hz : Eq (CochainComplex.HomComplex.δ n (HAdd.hAdd n 1) ↑z) 0
      ⊢ Eq (CochainComplex.HomComplex.δ n (HAdd.hAdd n 1) (HSMul.hSMul r ↑z)) 0
    -/
    simp only [δ_smul, hz, smul_zero]⟩
    /-
      🎉 no goals
    -/


@[simp]
                                                                  /-
                                                                    C : Type u
                                                                    inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                    inst✝ : CategoryTheory.Preadditive C
                                                                    F G : CochainComplex C Int
                                                                    n : Int
                                                                    ⊢ Eq (↑0) 0
                                                                  -/
lemma coe_zero : (↑(0 : Cocycle F G n) : Cochain F G n) = 0 := by rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
lemma coe_add (z₁ z₂ : Cocycle F G n) :
    (↑(z₁ + z₂) : Cochain F G n) = (z₁ : Cochain F G n) + (z₂ : Cochain F G n) := rfl


@[simp]
lemma coe_neg (z : Cocycle F G n) :
    (↑(-z) : Cochain F G n) = -(z : Cochain F G n) := rfl


@[simp]
lemma coe_smul (z : Cocycle F G n) (x : R) :
    (↑(x • z) : Cochain F G n) = x • (z : Cochain F G n) := rfl


@[simp]
lemma coe_units_smul (z : Cocycle F G n) (x : Rˣ) :
    (↑(x • z) : Cochain F G n) = x • (z : Cochain F G n) := rfl


@[simp]
lemma coe_sub (z₁ z₂ : Cocycle F G n) :
    (↑(z₁ - z₂) : Cochain F G n) = (z₁ : Cochain F G n) - (z₂ : Cochain F G n) := rfl


instance : Module R (Cocycle F G n) where
                   /-
                     C : Type u
                     inst✝³ : CategoryTheory.Category.{v, u} C
                     inst✝² : CategoryTheory.Preadditive C
                     R : Type u_1
                     inst✝¹ : Ring R
                     inst✝ : CategoryTheory.Linear R C
                     F G K L : CochainComplex C Int
                     n m : Int
                     x✝ : CochainComplex.HomComplex.Cocycle F G n
                     ⊢ Eq (HSMul.hSMul 1 x✝) x✝
                   -/
  one_smul _ := by aesop
                   /-
                     🎉 no goals
                   -/
                       /-
                         C : Type u
                         inst✝³ : CategoryTheory.Category.{v, u} C
                         inst✝² : CategoryTheory.Preadditive C
                         R : Type u_1
                         inst✝¹ : Ring R
                         inst✝ : CategoryTheory.Linear R C
                         F G K L : CochainComplex C Int
                         n m : Int
                         x✝² x✝¹ : R
                         x✝ : CochainComplex.HomComplex.Cocycle F G n
                         ⊢ Eq (HSMul.hSMul (HMul.hMul x✝² x✝¹) x✝) (HSMul.hSMul x✝² (HSMul.hSMul x✝¹ x✝))
                       -/
  mul_smul _ _ _ := by ext; dsimp; rw [smul_smul]
                                   /-
                                     🎉 no goals
                                   -/
                    /-
                      C : Type u
                      inst✝³ : CategoryTheory.Category.{v, u} C
                      inst✝² : CategoryTheory.Preadditive C
                      R : Type u_1
                      inst✝¹ : Ring R
                      inst✝ : CategoryTheory.Linear R C
                      F G K L : CochainComplex C Int
                      n m : Int
                      x✝ : R
                      ⊢ Eq (HSMul.hSMul x✝ 0) 0
                    -/
  smul_zero _ := by aesop
                    /-
                      🎉 no goals
                    -/
                       /-
                         C : Type u
                         inst✝³ : CategoryTheory.Category.{v, u} C
                         inst✝² : CategoryTheory.Preadditive C
                         R : Type u_1
                         inst✝¹ : Ring R
                         inst✝ : CategoryTheory.Linear R C
                         F G K L : CochainComplex C Int
                         n m : Int
                         x✝² : R
                         x✝¹ x✝ : CochainComplex.HomComplex.Cocycle F G n
                         ⊢ Eq (HSMul.hSMul x✝² (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd (HSMul.hSMul x✝² x✝¹) (HS …
                       -/
  smul_add _ _ _ := by aesop
                       /-
                         🎉 no goals
                       -/
                       /-
                         C : Type u
                         inst✝³ : CategoryTheory.Category.{v, u} C
                         inst✝² : CategoryTheory.Preadditive C
                         R : Type u_1
                         inst✝¹ : Ring R
                         inst✝ : CategoryTheory.Linear R C
                         F G K L : CochainComplex C Int
                         n m : Int
                         x✝² x✝¹ : R
                         x✝ : CochainComplex.HomComplex.Cocycle F G n
                         ⊢ Eq (HSMul.hSMul (HAdd.hAdd x✝² x✝¹) x✝) (HAdd.hAdd (HSMul.hSMul x✝² x✝) (HSM …
                       -/
  add_smul _ _ _ := by ext; dsimp; rw [add_smul]
                                   /-
                                     🎉 no goals
                                   -/
                  /-
                    C : Type u
                    inst✝³ : CategoryTheory.Category.{v, u} C
                    inst✝² : CategoryTheory.Preadditive C
                    R : Type u_1
                    inst✝¹ : Ring R
                    inst✝ : CategoryTheory.Linear R C
                    F G K L : CochainComplex C Int
                    n m : Int
                    ⊢ ∀ (x : CochainComplex.HomComplex.Cocycle F G n), Eq (HSMul.hSMul 0 x) 0
                  -/
  zero_smul := by aesop
                  /-
                    🎉 no goals
                  -/


/-- Constructor for `Cocycle F G n`, taking as inputs `z : Cochain F G n`, an integer
`m : ℤ` such that `n + 1 = m`, and the relation `δ n m z = 0`. -/
@[simps]
def mk (z : Cochain F G n) (m : ℤ) (hnm : n + 1 = m) (h : δ n m z = 0) : Cocycle F G n :=
         /-
           C : Type u
           inst✝³ : CategoryTheory.Category.{v, u} C
           inst✝² : CategoryTheory.Preadditive C
           R : Type u_1
           inst✝¹ : Ring R
           inst✝ : CategoryTheory.Linear R C
           F G K L : CochainComplex C Int
           n m✝ : Int
           z : CochainComplex.HomComplex.Cochain F G n
           m : Int
           hnm : Eq (HAdd.hAdd n 1) m
           h : Eq (CochainComplex.HomComplex.δ n m z) 0
           ⊢ Membership.mem (CochainComplex.HomComplex.cocycle F G n) z
         -/
  ⟨z, by simpa only [mem_iff n m hnm z] using h⟩
         /-
           🎉 no goals
         -/


@[simp]
lemma δ_eq_zero {n : ℤ} (z : Cocycle F G n) (m : ℤ) : δ n m (z : Cochain F G n) = 0 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    n : Int
    z : CochainComplex.HomComplex.Cocycle F G n
    m : Int
    ⊢ Eq (CochainComplex.HomComplex.δ n m ↑z) 0
  -/
  by_cases h : n + 1 = m
    /-
      case pos
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      n : Int
      z : CochainComplex.HomComplex.Cocycle F G n
      m : Int
      h : Eq (HAdd.hAdd n 1) m
      ⊢ Eq (CochainComplex.HomComplex.δ n m ↑z) 0
    -/
  · rw [← mem_iff n m h]
    /-
      case pos
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      n : Int
      z : CochainComplex.HomComplex.Cocycle F G n
      m : Int
      h : Eq (HAdd.hAdd n 1) m
      ⊢ Membership.mem (CochainComplex.HomComplex.cocycle F G n) ↑z
    -/
    exact z.2
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      n : Int
      z : CochainComplex.HomComplex.Cocycle F G n
      m : Int
      h : Not (Eq (HAdd.hAdd n 1) m)
      ⊢ Eq (CochainComplex.HomComplex.δ n m ↑z) 0
    -/
  · exact δ_shape n m h _
    /-
      🎉 no goals
    -/


/-- The `0`-cocycle associated to a morphism in `CochainComplex C ℤ`. -/
@[simps!]
                                                                                 /-
                                                                                   C : Type u
                                                                                   inst✝³ : CategoryTheory.Category.{v, u} C
                                                                                   inst✝² : CategoryTheory.Preadditive C
                                                                                   R : Type u_1
                                                                                   inst✝¹ : Ring R
                                                                                   inst✝ : CategoryTheory.Linear R C
                                                                                   F G K L : CochainComplex C Int
                                                                                   n m : Int
                                                                                   φ : Quiver.Hom F G
                                                                                   ⊢ Eq (CochainComplex.HomComplex.δ 0 1 (CochainComplex.HomComplex.Cochain.ofHom …
                                                                                 -/
def ofHom (φ : F ⟶ G) : Cocycle F G 0 := mk (Cochain.ofHom φ) 1 (zero_add 1) (by simp)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


/-- The morphism in `CochainComplex C ℤ` associated to a `0`-cocycle. -/
@[simps]
def homOf (z : Cocycle F G 0) : F ⟶ G where
  f i := (z : Cochain _ _ _).v i i (add_zero i)
  comm' := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CategoryTheory.Linear R C
      F G K L : CochainComplex C Int
      n m : Int
      z : CochainComplex.HomComplex.Cocycle F G 0
      ⊢ ∀ (i j : Int), (ComplexShape.up Int).Rel i j → Eq (CategoryTheory.CategorySt …
    -/
    rintro i j rfl
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CategoryTheory.Linear R C
      F G K L : CochainComplex C Int
      n m : Int
      z : CochainComplex.HomComplex.Cocycle F G 0
      i : Int
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => (↑z).v i i ⋯) i) (G.d i (H …
    -/
    rcases z with ⟨z, hz⟩
    /-
      case mk
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CategoryTheory.Linear R C
      F G K L : CochainComplex C Int
      n m i : Int
      z : CochainComplex.HomComplex.Cochain F G 0
      hz : Membership.mem (CochainComplex.HomComplex.cocycle F G 0) z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => (↑⟨z, hz⟩).v i i ⋯) i) (G. …
    -/
    dsimp
    /-
      case mk
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CategoryTheory.Linear R C
      F G K L : CochainComplex C Int
      n m i : Int
      z : CochainComplex.HomComplex.Cochain F G 0
      hz : Membership.mem (CochainComplex.HomComplex.cocycle F G 0) z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (z.v i i ⋯) (G.d i (HAdd.hAdd i 1)))  …
    -/
    rw [mem_iff 0 1 (zero_add 1)] at hz
    simpa only [δ_zero_cochain_v, Cochain.zero_v, sub_eq_zero]
      using Cochain.congr_v hz i (i + 1) rfl


@[simp]
                                                                  /-
                                                                    C : Type u
                                                                    inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                    inst✝ : CategoryTheory.Preadditive C
                                                                    F G : CochainComplex C Int
                                                                    φ : Quiver.Hom F G
                                                                    ⊢ Eq (CochainComplex.HomComplex.Cocycle.ofHom φ).homOf φ
                                                                  -/
lemma homOf_ofHom_eq_self (φ : F ⟶ G) : homOf (ofHom φ) = φ := by aesop_cat
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
                                                                          /-
                                                                            C : Type u
                                                                            inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                            inst✝ : CategoryTheory.Preadditive C
                                                                            F G : CochainComplex C Int
                                                                            z : CochainComplex.HomComplex.Cocycle F G 0
                                                                            ⊢ Eq (CochainComplex.HomComplex.Cocycle.ofHom z.homOf) z
                                                                          -/
lemma ofHom_homOf_eq_self (z : Cocycle F G 0) : ofHom (homOf z) = z := by aesop_cat
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
lemma cochain_ofHom_homOf_eq_coe (z : Cocycle F G 0) :
    Cochain.ofHom (homOf z) = (z : Cochain F G 0) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    z : CochainComplex.HomComplex.Cocycle F G 0
    ⊢ Eq (CochainComplex.HomComplex.Cochain.ofHom z.homOf) ↑z
  -/
  simpa only [Cocycle.ext_iff] using ofHom_homOf_eq_self z
  /-
    🎉 no goals
  -/


/-- The additive equivalence between morphisms in `CochainComplex C ℤ` and `0`-cocycles. -/
@[simps]
def equivHom : (F ⟶ G) ≃+ Cocycle F G 0 where
  toFun := ofHom
  invFun := homOf
  left_inv := homOf_ofHom_eq_self
  right_inv := ofHom_homOf_eq_self
                 /-
                   C : Type u
                   inst✝³ : CategoryTheory.Category.{v, u} C
                   inst✝² : CategoryTheory.Preadditive C
                   R : Type u_1
                   inst✝¹ : Ring R
                   inst✝ : CategoryTheory.Linear R C
                   F G K L : CochainComplex C Int
                   n m : Int
                   ⊢ ∀ (x y : Quiver.Hom F G), Eq ({ toFun := CochainComplex.HomComplex.Cocycle.o …
                 -/
  map_add' := by aesop_cat
                 /-
                   🎉 no goals
                 -/


/-- The `1`-cocycle given by the differential on a cochain complex. -/
@[simps!]
def diff : Cocycle K K 1 :=
  Cocycle.mk (Cochain.diff K) 2 rfl (by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CategoryTheory.Linear R C
      F G K L : CochainComplex C Int
      n m : Int
      ⊢ Eq (CochainComplex.HomComplex.δ 1 2 (CochainComplex.HomComplex.Cochain.diff  …
    -/
    ext p q hpq
    simp only [Cochain.zero_v, δ_v 1 2 rfl _ p q hpq _ _ rfl rfl, Cochain.diff_v,
      HomologicalComplex.d_comp_d, smul_zero, add_zero])


@[simp]
lemma δ_comp_zero_cocycle {n : ℤ} (z₁ : Cochain F G n) (z₂ : Cocycle G K 0) (m : ℤ) :
    δ n m (z₁.comp z₂.1 (add_zero n)) =
      (δ n m z₁).comp z₂.1 (add_zero m) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n
    z₂ : CochainComplex.HomComplex.Cocycle G K 0
    m : Int
    ⊢ Eq (CochainComplex.HomComplex.δ n m (z₁.comp ↑z₂ ⋯)) ((CochainComplex.HomCom …
  -/
  by_cases hnm : n + 1 = m
    /-
      case pos
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      F G K : CochainComplex C Int
      n : Int
      z₁ : CochainComplex.HomComplex.Cochain F G n
      z₂ : CochainComplex.HomComplex.Cocycle G K 0
      m : Int
      hnm : Eq (HAdd.hAdd n 1) m
      ⊢ Eq (CochainComplex.HomComplex.δ n m (z₁.comp ↑z₂ ⋯)) ((CochainComplex.HomCom …
    -/
  · simp [δ_comp_zero_cochain _ _ _ hnm]
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      F G K : CochainComplex C Int
      n : Int
      z₁ : CochainComplex.HomComplex.Cochain F G n
      z₂ : CochainComplex.HomComplex.Cocycle G K 0
      m : Int
      hnm : Not (Eq (HAdd.hAdd n 1) m)
      ⊢ Eq (CochainComplex.HomComplex.δ n m (z₁.comp ↑z₂ ⋯)) ((CochainComplex.HomCom …
    -/
  · simp [δ_shape _ _ hnm]
    /-
      🎉 no goals
    -/


@[simp]
lemma δ_comp_ofHom {n : ℤ} (z₁ : Cochain F G n) (f : G ⟶ K) (m : ℤ) :
    δ n m (z₁.comp (Cochain.ofHom f) (add_zero n)) =
      (δ n m z₁).comp (Cochain.ofHom f) (add_zero m) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n
    f : Quiver.Hom G K
    m : Int
    ⊢ Eq (CochainComplex.HomComplex.δ n m (z₁.comp (CochainComplex.HomComplex.Coch …
  -/
  rw [← Cocycle.ofHom_coe, δ_comp_zero_cocycle]
  /-
    🎉 no goals
  -/



@[simp]
lemma δ_zero_cocycle_comp {n : ℤ} (z₁ : Cocycle F G 0) (z₂ : Cochain G K n) (m : ℤ) :
    δ n m (z₁.1.comp z₂ (zero_add n)) =
      z₁.1.comp (δ n m z₂) (zero_add m) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n : Int
    z₁ : CochainComplex.HomComplex.Cocycle F G 0
    z₂ : CochainComplex.HomComplex.Cochain G K n
    m : Int
    ⊢ Eq (CochainComplex.HomComplex.δ n m ((↑z₁).comp z₂ ⋯)) ((↑z₁).comp (CochainC …
  -/
  by_cases hnm : n + 1 = m
    /-
      case pos
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      F G K : CochainComplex C Int
      n : Int
      z₁ : CochainComplex.HomComplex.Cocycle F G 0
      z₂ : CochainComplex.HomComplex.Cochain G K n
      m : Int
      hnm : Eq (HAdd.hAdd n 1) m
      ⊢ Eq (CochainComplex.HomComplex.δ n m ((↑z₁).comp z₂ ⋯)) ((↑z₁).comp (CochainC …
    -/
  · simp [δ_zero_cochain_comp _ _ _ hnm]
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Preadditive C
      F G K : CochainComplex C Int
      n : Int
      z₁ : CochainComplex.HomComplex.Cocycle F G 0
      z₂ : CochainComplex.HomComplex.Cochain G K n
      m : Int
      hnm : Not (Eq (HAdd.hAdd n 1) m)
      ⊢ Eq (CochainComplex.HomComplex.δ n m ((↑z₁).comp z₂ ⋯)) ((↑z₁).comp (CochainC …
    -/
  · simp [δ_shape _ _ hnm]
    /-
      🎉 no goals
    -/


@[simp]
lemma δ_ofHom_comp {n : ℤ} (f : F ⟶ G) (z : Cochain G K n) (m : ℤ) :
    δ n m ((Cochain.ofHom f).comp z (zero_add n)) =
      (Cochain.ofHom f).comp (δ n m z) (zero_add m) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    n : Int
    f : Quiver.Hom F G
    z : CochainComplex.HomComplex.Cochain G K n
    m : Int
    ⊢ Eq (CochainComplex.HomComplex.δ n m ((CochainComplex.HomComplex.Cochain.ofHo …
  -/
  rw [← Cocycle.ofHom_coe, δ_zero_cocycle_comp]
  /-
    🎉 no goals
  -/


/-- Given two morphisms of complexes `φ₁ φ₂ : F ⟶ G`, the datum of an homotopy between `φ₁` and
`φ₂` is equivalent to the datum of a `1`-cochain `z` such that `δ (-1) 0 z` is the difference
of the zero cochains associated to `φ₂` and `φ₁`. -/
@[simps]
def equivHomotopy (φ₁ φ₂ : F ⟶ G) :
    Homotopy φ₁ φ₂ ≃
      { z : Cochain F G (-1) // Cochain.ofHom φ₁ = δ (-1) 0 z + Cochain.ofHom φ₂ } where
                                         /-
                                           C : Type u
                                           inst✝³ : CategoryTheory.Category.{v, u} C
                                           inst✝² : CategoryTheory.Preadditive C
                                           R : Type u_1
                                           inst✝¹ : Ring R
                                           inst✝ : CategoryTheory.Linear R C
                                           F G K L : CochainComplex C Int
                                           n m : Int
                                           φ₁ φ₂ : Quiver.Hom F G
                                           ho : Homotopy φ₁ φ₂
                                           ⊢ Eq (CochainComplex.HomComplex.Cochain.ofHom φ₁) (HAdd.hAdd (CochainComplex.H …
                                         -/
  toFun ho := ⟨Cochain.ofHomotopy ho, by simp only [δ_ofHomotopy, sub_add_cancel]⟩
                                         /-
                                           🎉 no goals
                                         -/
  invFun z :=
    { hom := fun i j => if hij : i + (-1) = j then z.1.v i j hij else 0
                                                                     /-
                                                                       C : Type u
                                                                       inst✝³ : CategoryTheory.Category.{v, u} C
                                                                       inst✝² : CategoryTheory.Preadditive C
                                                                       R : Type u_1
                                                                       inst✝¹ : Ring R
                                                                       inst✝ : CategoryTheory.Linear R C
                                                                       F G K L : CochainComplex C Int
                                                                       n m : Int
                                                                       φ₁ φ₂ : Quiver.Hom F G
                                                                       z : Subtype fun z => Eq (CochainComplex.HomComplex.Cochain.ofHom φ₁) (HAdd.hAd …
                                                                       i j : Int
                                                                       hij : Ne (HAdd.hAdd j 1) i
                                                                       x✝ : Eq (HAdd.hAdd i (-1)) j
                                                                       ⊢ Eq (HAdd.hAdd j 1) i
                                                                     -/
      zero := fun i j (hij : j + 1 ≠ i) => dif_neg (fun _ => hij (by omega))
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
      comm := fun p => by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          R : Type u_1
          inst✝¹ : Ring R
          inst✝ : CategoryTheory.Linear R C
          F G K L : CochainComplex C Int
          n m : Int
          φ₁ φ₂ : Quiver.Hom F G
          z : Subtype fun z => Eq (CochainComplex.HomComplex.Cochain.ofHom φ₁) (HAdd.hAd …
          p : Int
          ⊢ Eq (φ₁.f p) (HAdd.hAdd (HAdd.hAdd ((dNext p) fun i j => dite (Eq (HAdd.hAdd  …
        -/
        have eq := Cochain.congr_v z.2 p p (add_zero p)
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          R : Type u_1
          inst✝¹ : Ring R
          inst✝ : CategoryTheory.Linear R C
          F G K L : CochainComplex C Int
          n m : Int
          φ₁ φ₂ : Quiver.Hom F G
          z : Subtype fun z => Eq (CochainComplex.HomComplex.Cochain.ofHom φ₁) (HAdd.hAd …
          p : Int
          eq : Eq ((CochainComplex.HomComplex.Cochain.ofHom φ₁).v p p ⋯) ((HAdd.hAdd (Co …
          ⊢ Eq (φ₁.f p) (HAdd.hAdd (HAdd.hAdd ((dNext p) fun i j => dite (Eq (HAdd.hAdd  …
        -/
        have h₁ : (ComplexShape.up ℤ).Rel (p - 1) p := by simp
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          R : Type u_1
          inst✝¹ : Ring R
          inst✝ : CategoryTheory.Linear R C
          F G K L : CochainComplex C Int
          n m : Int
          φ₁ φ₂ : Quiver.Hom F G
          z : Subtype fun z => Eq (CochainComplex.HomComplex.Cochain.ofHom φ₁) (HAdd.hAd …
          p : Int
          eq : Eq ((CochainComplex.HomComplex.Cochain.ofHom φ₁).v p p ⋯) ((HAdd.hAdd (Co …
          h₁ : (ComplexShape.up Int).Rel (HSub.hSub p 1) p
          ⊢ Eq (φ₁.f p) (HAdd.hAdd (HAdd.hAdd ((dNext p) fun i j => dite (Eq (HAdd.hAdd  …
        -/
        have h₂ : (ComplexShape.up ℤ).Rel p (p + 1) := by simp
        simp only [δ_neg_one_cochain, Cochain.ofHom_v, ComplexShape.up_Rel, Cochain.add_v,
          Homotopy.nullHomotopicMap'_f h₁ h₂] at eq
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          R : Type u_1
          inst✝¹ : Ring R
          inst✝ : CategoryTheory.Linear R C
          F G K L : CochainComplex C Int
          n m : Int
          φ₁ φ₂ : Quiver.Hom F G
          z : Subtype fun z => Eq (CochainComplex.HomComplex.Cochain.ofHom φ₁) (HAdd.hAd …
          p : Int
          h₁ : (ComplexShape.up Int).Rel (HSub.hSub p 1) p
          h₂ : (ComplexShape.up Int).Rel p (HAdd.hAdd p 1)
          eq : Eq (φ₁.f p) (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (F. …
          ⊢ Eq (φ₁.f p) (HAdd.hAdd (HAdd.hAdd ((dNext p) fun i j => dite (Eq (HAdd.hAdd  …
        -/
        rw [dNext_eq _ h₂, prevD_eq _ h₁, eq, dif_pos, dif_pos] }
        /-
          🎉 no goals
        -/
  left_inv := fun ho => by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CategoryTheory.Linear R C
      F G K L : CochainComplex C Int
      n m : Int
      φ₁ φ₂ : Quiver.Hom F G
      ho : Homotopy φ₁ φ₂
      ⊢ Eq ((fun z => { hom := fun i j => dite (Eq (HAdd.hAdd i (-1)) j) (fun hij => …
    -/
    ext i j
    /-
      case hom.h.h
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CategoryTheory.Linear R C
      F G K L : CochainComplex C Int
      n m : Int
      φ₁ φ₂ : Quiver.Hom F G
      ho : Homotopy φ₁ φ₂
      i j : Int
      ⊢ Eq (((fun z => { hom := fun i j => dite (Eq (HAdd.hAdd i (-1)) j) (fun hij = …
    -/
    dsimp
    /-
      case hom.h.h
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CategoryTheory.Linear R C
      F G K L : CochainComplex C Int
      n m : Int
      φ₁ φ₂ : Quiver.Hom F G
      ho : Homotopy φ₁ φ₂
      i j : Int
      ⊢ Eq (dite (Eq (HAdd.hAdd i (-1)) j) (fun hij => (CochainComplex.HomComplex.Co …
    -/
    split_ifs with h
      /-
        case pos
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Preadditive C
        R : Type u_1
        inst✝¹ : Ring R
        inst✝ : CategoryTheory.Linear R C
        F G K L : CochainComplex C Int
        n m : Int
        φ₁ φ₂ : Quiver.Hom F G
        ho : Homotopy φ₁ φ₂
        i j : Int
        h : Eq (HAdd.hAdd i (-1)) j
        ⊢ Eq ((CochainComplex.HomComplex.Cochain.ofHomotopy ho).v i j h) (ho.hom i j)
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Preadditive C
        R : Type u_1
        inst✝¹ : Ring R
        inst✝ : CategoryTheory.Linear R C
        F G K L : CochainComplex C Int
        n m : Int
        φ₁ φ₂ : Quiver.Hom F G
        ho : Homotopy φ₁ φ₂
        i j : Int
        h : Not (Eq (HAdd.hAdd i (-1)) j)
        ⊢ Eq 0 (ho.hom i j)
      -/
    · rw [ho.zero i j (fun h' => h (by dsimp at h'; omega))]
      /-
        🎉 no goals
      -/
  right_inv := fun z => by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CategoryTheory.Linear R C
      F G K L : CochainComplex C Int
      n m : Int
      φ₁ φ₂ : Quiver.Hom F G
      z : Subtype fun z => Eq (CochainComplex.HomComplex.Cochain.ofHom φ₁) (HAdd.hAd …
      ⊢ Eq ((fun ho => ⟨CochainComplex.HomComplex.Cochain.ofHomotopy ho, ⋯⟩) ((fun z …
    -/
    ext p q hpq
    /-
      case a.h
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CategoryTheory.Linear R C
      F G K L : CochainComplex C Int
      n m : Int
      φ₁ φ₂ : Quiver.Hom F G
      z : Subtype fun z => Eq (CochainComplex.HomComplex.Cochain.ofHom φ₁) (HAdd.hAd …
      p q : Int
      hpq : Eq (HAdd.hAdd p (-1)) q
      ⊢ Eq ((↑((fun ho => ⟨CochainComplex.HomComplex.Cochain.ofHomotopy ho, ⋯⟩) ((fu …
    -/
    dsimp [Cochain.ofHomotopy]
    /-
      case a.h
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Preadditive C
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CategoryTheory.Linear R C
      F G K L : CochainComplex C Int
      n m : Int
      φ₁ φ₂ : Quiver.Hom F G
      z : Subtype fun z => Eq (CochainComplex.HomComplex.Cochain.ofHom φ₁) (HAdd.hAd …
      p q : Int
      hpq : Eq (HAdd.hAdd p (-1)) q
      ⊢ Eq (dite (Eq (HAdd.hAdd p (-1)) q) (fun hij => (↑z).v p q hij) fun hij => 0) …
    -/
    rw [dif_pos hpq]
    /-
      🎉 no goals
    -/


@[simp]
lemma equivHomotopy_apply_of_eq {φ₁ φ₂ : F ⟶ G} (h : φ₁ = φ₂) :
    (equivHomotopy _ _ (Homotopy.ofEq h)).1 = 0 := rfl


lemma ofHom_injective {f₁ f₂ : F ⟶ G} (h : ofHom f₁ = ofHom f₂) : f₁ = f₂ :=
                                       /-
                                         C : Type u
                                         inst✝¹ : CategoryTheory.Category.{v, u} C
                                         inst✝ : CategoryTheory.Preadditive C
                                         F G : CochainComplex C Int
                                         f₁ f₂ : Quiver.Hom F G
                                         h : Eq (CochainComplex.HomComplex.Cochain.ofHom f₁) (CochainComplex.HomComplex …
                                         ⊢ Eq ((CochainComplex.HomComplex.Cocycle.equivHom F G) f₁) ((CochainComplex.Ho …
                                       -/
  (Cocycle.equivHom F G).injective (by ext1; exact h)
                                             /-
                                               🎉 no goals
                                             -/


/-- If `Φ : C ⥤ D` is an additive functor, a cochain `z : Cochain K L n` between
cochain complexes in `C` can be mapped to a cochain between the cochain complexes
in `D` obtained by applying the functor
`Φ.mapHomologicalComplex _ : CochainComplex C ℤ ⥤ CochainComplex D ℤ`. -/
def map : Cochain ((Φ.mapHomologicalComplex _).obj K) ((Φ.mapHomologicalComplex _).obj L) n :=
  Cochain.mk (fun p q hpq => Φ.map (z.v p q hpq))


@[simp]
lemma map_v (p q : ℤ) (hpq : p + n = q) : (z.map Φ).v p q hpq = Φ.map (z.v p q hpq) := rfl


@[simp]
                                                          /-
                                                            C : Type u
                                                            inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                            inst✝³ : CategoryTheory.Preadditive C
                                                            K L : CochainComplex C Int
                                                            n : Int
                                                            D : Type u_2
                                                            inst✝² : CategoryTheory.Category.{u_3, u_2} D
                                                            inst✝¹ : CategoryTheory.Preadditive D
                                                            z z' : CochainComplex.HomComplex.Cochain K L n
                                                            Φ : CategoryTheory.Functor C D
                                                            inst✝ : Φ.Additive
                                                            ⊢ Eq ((HAdd.hAdd z z').map Φ) (HAdd.hAdd (z.map Φ) (z'.map Φ))
                                                          -/
lemma map_add : (z + z').map Φ = z.map Φ + z'.map Φ := by aesop_cat
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
                                            /-
                                              C : Type u
                                              inst✝⁴ : CategoryTheory.Category.{v, u} C
                                              inst✝³ : CategoryTheory.Preadditive C
                                              K L : CochainComplex C Int
                                              n : Int
                                              D : Type u_2
                                              inst✝² : CategoryTheory.Category.{u_3, u_2} D
                                              inst✝¹ : CategoryTheory.Preadditive D
                                              z : CochainComplex.HomComplex.Cochain K L n
                                              Φ : CategoryTheory.Functor C D
                                              inst✝ : Φ.Additive
                                              ⊢ Eq ((Neg.neg z).map Φ) (Neg.neg (z.map Φ))
                                            -/
lemma map_neg : (-z).map Φ = -z.map Φ := by aesop_cat
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
                                                          /-
                                                            C : Type u
                                                            inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                            inst✝³ : CategoryTheory.Preadditive C
                                                            K L : CochainComplex C Int
                                                            n : Int
                                                            D : Type u_2
                                                            inst✝² : CategoryTheory.Category.{u_3, u_2} D
                                                            inst✝¹ : CategoryTheory.Preadditive D
                                                            z z' : CochainComplex.HomComplex.Cochain K L n
                                                            Φ : CategoryTheory.Functor C D
                                                            inst✝ : Φ.Additive
                                                            ⊢ Eq ((HSub.hSub z z').map Φ) (HSub.hSub (z.map Φ) (z'.map Φ))
                                                          -/
lemma map_sub : (z - z').map Φ = z.map Φ - z'.map Φ := by aesop_cat
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
                                                     /-
                                                       C : Type u
                                                       inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                       inst✝³ : CategoryTheory.Preadditive C
                                                       K L : CochainComplex C Int
                                                       n : Int
                                                       D : Type u_2
                                                       inst✝² : CategoryTheory.Category.{u_3, u_2} D
                                                       inst✝¹ : CategoryTheory.Preadditive D
                                                       Φ : CategoryTheory.Functor C D
                                                       inst✝ : Φ.Additive
                                                       ⊢ Eq (CochainComplex.HomComplex.Cochain.map 0 Φ) 0
                                                     -/
lemma map_zero : (0 : Cochain K L n).map Φ = 0 := by aesop_cat
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
lemma map_comp {n₁ n₂ n₁₂ : ℤ} (z₁ : Cochain F G n₁) (z₂ : Cochain G K n₂) (h : n₁ + n₂ = n₁₂)
    (Φ : C ⥤ D) [Φ.Additive] :
    (Cochain.comp z₁ z₂ h).map Φ = Cochain.comp (z₁.map Φ) (z₂.map Φ) h := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    inst✝¹ : CategoryTheory.Preadditive D
    n₁ n₂ n₁₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    Φ : CategoryTheory.Functor C D
    inst✝ : Φ.Additive
    ⊢ Eq ((z₁.comp z₂ h).map Φ) ((z₁.map Φ).comp (z₂.map Φ) h)
  -/
  ext p q hpq
  /-
    case h
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    inst✝¹ : CategoryTheory.Preadditive D
    n₁ n₂ n₁₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    Φ : CategoryTheory.Functor C D
    inst✝ : Φ.Additive
    p q : Int
    hpq : Eq (HAdd.hAdd p n₁₂) q
    ⊢ Eq (((z₁.comp z₂ h).map Φ).v p q hpq) (((z₁.map Φ).comp (z₂.map Φ) h).v p q  …
  -/
  dsimp
  /-
    case h
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Preadditive C
    F G K : CochainComplex C Int
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    inst✝¹ : CategoryTheory.Preadditive D
    n₁ n₂ n₁₂ : Int
    z₁ : CochainComplex.HomComplex.Cochain F G n₁
    z₂ : CochainComplex.HomComplex.Cochain G K n₂
    h : Eq (HAdd.hAdd n₁ n₂) n₁₂
    Φ : CategoryTheory.Functor C D
    inst✝ : Φ.Additive
    p q : Int
    hpq : Eq (HAdd.hAdd p n₁₂) q
    ⊢ Eq (Φ.map ((z₁.comp z₂ h).v p q hpq)) (((z₁.map Φ).comp (z₂.map Φ) h).v p q  …
  -/
  simp only [map_v, comp_v _ _ h p _ q rfl (by omega), Φ.map_comp]
  /-
    🎉 no goals
  -/


@[simp]
lemma map_ofHom :
                                                                                      /-
                                                                                        C : Type u
                                                                                        inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                                                        inst✝³ : CategoryTheory.Preadditive C
                                                                                        K L : CochainComplex C Int
                                                                                        D : Type u_2
                                                                                        inst✝² : CategoryTheory.Category.{u_3, u_2} D
                                                                                        inst✝¹ : CategoryTheory.Preadditive D
                                                                                        f : Quiver.Hom K L
                                                                                        Φ : CategoryTheory.Functor C D
                                                                                        inst✝ : Φ.Additive
                                                                                        ⊢ Eq ((CochainComplex.HomComplex.Cochain.ofHom f).map Φ) (CochainComplex.HomCo …
                                                                                      -/
    (Cochain.ofHom f).map Φ = Cochain.ofHom ((Φ.mapHomologicalComplex _).map f) := by aesop_cat
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


@[simp]
lemma δ_map : δ n m (z.map Φ) = (δ n m z).map Φ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Preadditive C
    K L : CochainComplex C Int
    n m : Int
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    inst✝¹ : CategoryTheory.Preadditive D
    z : CochainComplex.HomComplex.Cochain K L n
    Φ : CategoryTheory.Functor C D
    inst✝ : Φ.Additive
    ⊢ Eq (CochainComplex.HomComplex.δ n m (z.map Φ)) ((CochainComplex.HomComplex.δ …
  -/
  by_cases hnm : n + 1 = m
    /-
      case pos
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Preadditive C
      K L : CochainComplex C Int
      n m : Int
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      inst✝¹ : CategoryTheory.Preadditive D
      z : CochainComplex.HomComplex.Cochain K L n
      Φ : CategoryTheory.Functor C D
      inst✝ : Φ.Additive
      hnm : Eq (HAdd.hAdd n 1) m
      ⊢ Eq (CochainComplex.HomComplex.δ n m (z.map Φ)) ((CochainComplex.HomComplex.δ …
    -/
  · ext p q hpq
    /-
      case pos.h
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Preadditive C
      K L : CochainComplex C Int
      n m : Int
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      inst✝¹ : CategoryTheory.Preadditive D
      z : CochainComplex.HomComplex.Cochain K L n
      Φ : CategoryTheory.Functor C D
      inst✝ : Φ.Additive
      hnm : Eq (HAdd.hAdd n 1) m
      p q : Int
      hpq : Eq (HAdd.hAdd p m) q
      ⊢ Eq ((CochainComplex.HomComplex.δ n m (z.map Φ)).v p q hpq) (((CochainComplex …
    -/
    dsimp
    simp only [δ_v n m hnm _ p q hpq (q-1) (p+1) rfl rfl,
      Functor.map_add, Functor.map_comp, Functor.map_units_smul,
      Cochain.map_v, Functor.mapHomologicalComplex_obj_d]
    /-
      case neg
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Preadditive C
      K L : CochainComplex C Int
      n m : Int
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      inst✝¹ : CategoryTheory.Preadditive D
      z : CochainComplex.HomComplex.Cochain K L n
      Φ : CategoryTheory.Functor C D
      inst✝ : Φ.Additive
      hnm : Not (Eq (HAdd.hAdd n 1) m)
      ⊢ Eq (CochainComplex.HomComplex.δ n m (z.map Φ)) ((CochainComplex.HomComplex.δ …
    -/
  · simp only [δ_shape _ _ hnm, Cochain.map_zero]
    /-
      🎉 no goals
    -/



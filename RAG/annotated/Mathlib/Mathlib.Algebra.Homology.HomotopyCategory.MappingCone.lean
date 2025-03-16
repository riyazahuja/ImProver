instance [∀ p, HasBinaryBiproduct (F.X (p + 1)) (G.X p)] :
    HasHomotopyCofiber φ where
  hasBinaryBiproduct := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.268, u_2} D
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Preadditive D
      ι : Type u_3
      inst✝² : AddRightCancelSemigroup ι
      inst✝¹ : One ι
      F G : CochainComplex C ι
      φ : Quiver.Hom F G
      inst✝ : ∀ (p : ι), CategoryTheory.Limits.HasBinaryBiproduct (F.X (HAdd.hAdd p  …
      ⊢ ∀ (i j : ι), (ComplexShape.up ι).Rel i j → CategoryTheory.Limits.HasBinaryBi …
    -/
    rintro i _ rfl
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.268, u_2} D
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Preadditive D
      ι : Type u_3
      inst✝² : AddRightCancelSemigroup ι
      inst✝¹ : One ι
      F G : CochainComplex C ι
      φ : Quiver.Hom F G
      inst✝ : ∀ (p : ι), CategoryTheory.Limits.HasBinaryBiproduct (F.X (HAdd.hAdd p  …
      i : ι
      ⊢ CategoryTheory.Limits.HasBinaryBiproduct (F.X (HAdd.hAdd i 1)) (G.X i)
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/-- The mapping cone of a morphism of cochain complexes indexed by `ℤ`. -/
noncomputable def mappingCone := homotopyCofiber φ


/-- The left inclusion in the mapping cone, as a cochain of degree `-1`. -/
noncomputable def inl : Cochain F (mappingCone φ) (-1) :=
                                                            /-
                                                              C : Type u_1
                                                              D : Type u_2
                                                              inst✝⁴ : CategoryTheory.Category.{?u.3778, u_1} C
                                                              inst✝³ : CategoryTheory.Category.{?u.3782, u_2} D
                                                              inst✝² : CategoryTheory.Preadditive C
                                                              inst✝¹ : CategoryTheory.Preadditive D
                                                              F G : CochainComplex C Int
                                                              φ : Quiver.Hom F G
                                                              inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                                              p q : Int
                                                              hpq : Eq (HAdd.hAdd p (-1)) q
                                                              ⊢ (ComplexShape.up Int).Rel q p
                                                            -/
  Cochain.mk (fun p q hpq => homotopyCofiber.inlX φ p q (by dsimp; omega))
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- The right inclusion in the mapping cone. -/
noncomputable def inr : G ⟶ mappingCone φ := homotopyCofiber.inr φ


/-- The first projection from the mapping cone, as a cocyle of degree `1`. -/
noncomputable def fst : Cocycle (mappingCone φ) F 1 :=
                                                                                /-
                                                                                  C : Type u_1
                                                                                  D : Type u_2
                                                                                  inst✝⁴ : CategoryTheory.Category.{?u.6708, u_1} C
                                                                                  inst✝³ : CategoryTheory.Category.{?u.6712, u_2} D
                                                                                  inst✝² : CategoryTheory.Preadditive C
                                                                                  inst✝¹ : CategoryTheory.Preadditive D
                                                                                  F G : CochainComplex C Int
                                                                                  φ : Quiver.Hom F G
                                                                                  inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                                                                  ⊢ Eq (HAdd.hAdd 1 1) 2
                                                                                -/
  Cocycle.mk (Cochain.mk (fun p q hpq => homotopyCofiber.fstX φ p q hpq)) 2 (by omega) (by
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.6708, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.6712, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Preadditive D
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      ⊢ Eq (CochainComplex.HomComplex.δ 1 2 (CochainComplex.HomComplex.Cochain.mk fu …
    -/
    ext p _ rfl
    simp [δ_v 1 2 (by omega) _ p (p + 2) (by omega) (p + 1) (p + 1) (by omega) rfl,
      homotopyCofiber.d_fstX φ p (p + 1) (p + 2) rfl, mappingCone,
      show Int.negOnePow 2 = 1 by rfl])


/-- The second projection from the mapping cone, as a cochain of degree `0`. -/
noncomputable def snd : Cochain (mappingCone φ) G 0 :=
  Cochain.ofHoms (homotopyCofiber.sndX φ)


@[reassoc (attr := simp)]
lemma inl_v_fst_v (p q : ℤ) (hpq : q + 1 = p) :
                      /-
                        C : Type u_1
                        D : Type u_2
                        inst✝⁴ : CategoryTheory.Category.{?u.16507, u_1} C
                        inst✝³ : CategoryTheory.Category.{?u.16511, u_2} D
                        inst✝² : CategoryTheory.Preadditive C
                        inst✝¹ : CategoryTheory.Preadditive D
                        F G : CochainComplex C Int
                        φ : Quiver.Hom F G
                        inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                        p q : Int
                        hpq : Eq (HAdd.hAdd q 1) p
                        ⊢ Eq (HAdd.hAdd p (-1)) q
                      -/
    (inl φ).v p q (by rw [← hpq, add_neg_cancel_right]) ≫
                      /-
                        🎉 no goals
                      -/
      (fst φ : Cochain (mappingCone φ) F 1).v q p hpq = 𝟙 _ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    p q : Int
    hpq : Eq (HAdd.hAdd q 1) p
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inl φ).v …
  -/
  simp [inl, fst]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inl_v_snd_v (p q : ℤ) (hpq : p + (-1) = q) :
    (inl φ).v p q hpq ≫ (snd φ).v q q (add_zero q) = 0 := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    p q : Int
    hpq : Eq (HAdd.hAdd p (-1)) q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inl φ).v …
  -/
  simp [inl, snd]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inr_f_fst_v (p q : ℤ) (hpq : p + 1 = q) :
    (inr φ).f p ≫ (fst φ).1.v p q hpq = 0 := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    p q : Int
    hpq : Eq (HAdd.hAdd p 1) q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inr φ).f …
  -/
  simp [inr, fst]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inr_f_snd_v (p : ℤ) :
    (inr φ).f p ≫ (snd φ).v p p (add_zero p) = 𝟙 _ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    p : Int
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inr φ).f …
  -/
  simp [inr, snd]
  /-
    🎉 no goals
  -/


@[simp]
lemma inl_fst :
    (inl φ).comp (fst φ).1 (neg_add_cancel 1) = Cochain.ofHom (𝟙 F) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    ⊢ Eq ((CochainComplex.mappingCone.inl φ).comp ↑(CochainComplex.mappingCone.fst …
  -/
  ext p
  /-
    case h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    p : Int
    ⊢ Eq (((CochainComplex.mappingCone.inl φ).comp ↑(CochainComplex.mappingCone.fs …
  -/
  simp [Cochain.comp_v _ _ (neg_add_cancel 1) p (p-1) p rfl (by omega)]
  /-
    🎉 no goals
  -/


@[simp]
lemma inl_snd :
    (inl φ).comp (snd φ) (add_zero (-1)) = 0 := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    ⊢ Eq ((CochainComplex.mappingCone.inl φ).comp (CochainComplex.mappingCone.snd  …
  -/
  ext p q hpq
  /-
    case h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    p q : Int
    hpq : Eq (HAdd.hAdd p (-1)) q
    ⊢ Eq (((CochainComplex.mappingCone.inl φ).comp (CochainComplex.mappingCone.snd …
  -/
  simp [Cochain.comp_v _ _ (add_zero (-1)) p q q (by omega) (by omega)]
  /-
    🎉 no goals
  -/


@[simp]
lemma inr_fst :
    (Cochain.ofHom (inr φ)).comp (fst φ).1 (zero_add 1) = 0 := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.ofHom (CochainComplex.mappingCone.inr …
  -/
  ext p q hpq
  /-
    case h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    p q : Int
    hpq : Eq (HAdd.hAdd p 1) q
    ⊢ Eq (((CochainComplex.HomComplex.Cochain.ofHom (CochainComplex.mappingCone.in …
  -/
  simp [Cochain.comp_v _ _ (zero_add 1) p p q (by omega) (by omega)]
  /-
    🎉 no goals
  -/


@[simp]
lemma inr_snd :
                                                                                  /-
                                                                                    C : Type u_1
                                                                                    inst✝² : CategoryTheory.Category.{u_3, u_1} C
                                                                                    inst✝¹ : CategoryTheory.Preadditive C
                                                                                    F G : CochainComplex C Int
                                                                                    φ : Quiver.Hom F G
                                                                                    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                                                                    ⊢ Eq ((CochainComplex.HomComplex.Cochain.ofHom (CochainComplex.mappingCone.inr …
                                                                                  -/
    (Cochain.ofHom (inr φ)).comp (snd φ) (zero_add 0) = Cochain.ofHom (𝟙 G) := by aesop_cat
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[simp]
lemma inl_fst_assoc {K : CochainComplex C ℤ} {d e : ℤ} (γ : Cochain F K d) (he : 1 + d = e) :
                                           /-
                                             C : Type u_1
                                             D : Type u_2
                                             inst✝⁴ : CategoryTheory.Category.{?u.39496, u_1} C
                                             inst✝³ : CategoryTheory.Category.{?u.39500, u_2} D
                                             inst✝² : CategoryTheory.Preadditive C
                                             inst✝¹ : CategoryTheory.Preadditive D
                                             F G : CochainComplex C Int
                                             φ : Quiver.Hom F G
                                             inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                             K : CochainComplex C Int
                                             d e : Int
                                             γ : CochainComplex.HomComplex.Cochain F K d
                                             he : Eq (HAdd.hAdd 1 d) e
                                             ⊢ Eq (HAdd.hAdd (-1) e) d
                                           -/
    (inl φ).comp ((fst φ).1.comp γ he) (by rw [← he, neg_add_cancel_left]) = γ := by
                                           /-
                                             🎉 no goals
                                           -/
  rw [← Cochain.comp_assoc _ _ _ (neg_add_cancel 1) (by omega) (by omega), inl_fst,
    Cochain.id_comp]


@[simp]
lemma inl_snd_assoc {K : CochainComplex C ℤ} {d e f : ℤ} (γ : Cochain G K d)
    (he : 0 + d = e) (hf : -1 + e = f) :
    (inl φ).comp ((snd φ).comp γ he) hf = 0 := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    d e f : Int
    γ : CochainComplex.HomComplex.Cochain G K d
    he : Eq (HAdd.hAdd 0 d) e
    hf : Eq (HAdd.hAdd (-1) e) f
    ⊢ Eq ((CochainComplex.mappingCone.inl φ).comp ((CochainComplex.mappingCone.snd …
  -/
  obtain rfl : e = d := by omega
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    e f : Int
    hf : Eq (HAdd.hAdd (-1) e) f
    γ : CochainComplex.HomComplex.Cochain G K e
    he : Eq (HAdd.hAdd 0 e) e
    ⊢ Eq ((CochainComplex.mappingCone.inl φ).comp ((CochainComplex.mappingCone.snd …
  -/
  rw [← Cochain.comp_assoc_of_second_is_zero_cochain, inl_snd, Cochain.zero_comp]
  /-
    🎉 no goals
  -/


@[simp]
lemma inr_fst_assoc {K : CochainComplex C ℤ} {d e f : ℤ} (γ : Cochain F K d)
    (he : 1 + d = e) (hf : 0 + e = f) :
    (Cochain.ofHom (inr φ)).comp ((fst φ).1.comp γ he) hf = 0 := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    d e f : Int
    γ : CochainComplex.HomComplex.Cochain F K d
    he : Eq (HAdd.hAdd 1 d) e
    hf : Eq (HAdd.hAdd 0 e) f
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.ofHom (CochainComplex.mappingCone.inr …
  -/
  obtain rfl : e = f := by omega
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    d e : Int
    γ : CochainComplex.HomComplex.Cochain F K d
    he : Eq (HAdd.hAdd 1 d) e
    hf : Eq (HAdd.hAdd 0 e) e
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.ofHom (CochainComplex.mappingCone.inr …
  -/
  rw [← Cochain.comp_assoc_of_first_is_zero_cochain, inr_fst, Cochain.zero_comp]
  /-
    🎉 no goals
  -/


@[simp]
lemma inr_snd_assoc {K : CochainComplex C ℤ} {d e : ℤ} (γ : Cochain G K d) (he : 0 + d = e) :
                                                         /-
                                                           C : Type u_1
                                                           D : Type u_2
                                                           inst✝⁴ : CategoryTheory.Category.{?u.48781, u_1} C
                                                           inst✝³ : CategoryTheory.Category.{?u.48785, u_2} D
                                                           inst✝² : CategoryTheory.Preadditive C
                                                           inst✝¹ : CategoryTheory.Preadditive D
                                                           F G : CochainComplex C Int
                                                           φ : Quiver.Hom F G
                                                           inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                                           K : CochainComplex C Int
                                                           d e : Int
                                                           γ : CochainComplex.HomComplex.Cochain G K d
                                                           he : Eq (HAdd.hAdd 0 d) e
                                                           ⊢ Eq (HAdd.hAdd 0 e) d
                                                         -/
    (Cochain.ofHom (inr φ)).comp ((snd φ).comp γ he) (by simp only [← he, zero_add]) = γ := by
                                                         /-
                                                           🎉 no goals
                                                         -/
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    d e : Int
    γ : CochainComplex.HomComplex.Cochain G K d
    he : Eq (HAdd.hAdd 0 d) e
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.ofHom (CochainComplex.mappingCone.inr …
  -/
  obtain rfl : d = e := by omega
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    d : Int
    γ : CochainComplex.HomComplex.Cochain G K d
    he : Eq (HAdd.hAdd 0 d) d
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.ofHom (CochainComplex.mappingCone.inr …
  -/
  rw [← Cochain.comp_assoc_of_first_is_zero_cochain, inr_snd, Cochain.id_comp]
  /-
    🎉 no goals
  -/


lemma ext_to (i j : ℤ) (hij : i + 1 = j) {A : C} {f g : A ⟶ (mappingCone φ).X i}
    (h₁ : f ≫ (fst φ).1.v i j hij = g ≫ (fst φ).1.v i j hij)
    (h₂ : f ≫ (snd φ).v i i (add_zero i) = g ≫ (snd φ).v i i (add_zero i)) :
    f = g :=
                                            /-
                                              C : Type u_1
                                              inst✝² : CategoryTheory.Category.{u_3, u_1} C
                                              inst✝¹ : CategoryTheory.Preadditive C
                                              F G : CochainComplex C Int
                                              φ : Quiver.Hom F G
                                              inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                              i j : Int
                                              hij : Eq (HAdd.hAdd i 1) j
                                              A : C
                                              f g : Quiver.Hom A ((CochainComplex.mappingCone φ).X i)
                                              h₁ : Eq (CategoryTheory.CategoryStruct.comp f ((↑(CochainComplex.mappingCone.f …
                                              h₂ : Eq (CategoryTheory.CategoryStruct.comp f ((CochainComplex.mappingCone.snd …
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp f (HomologicalComplex.homotopyCofiber …
                                            -/
  homotopyCofiber.ext_to_X φ i j hij h₁ (by simpa [snd] using h₂)
                                            /-
                                              🎉 no goals
                                            -/


lemma ext_to_iff (i j : ℤ) (hij : i + 1 = j) {A : C} (f g : A ⟶ (mappingCone φ).X i) :
    f = g ↔ f ≫ (fst φ).1.v i j hij = g ≫ (fst φ).1.v i j hij ∧
      f ≫ (snd φ).v i i (add_zero i) = g ≫ (snd φ).v i i (add_zero i) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    i j : Int
    hij : Eq (HAdd.hAdd i 1) j
    A : C
    f g : Quiver.Hom A ((CochainComplex.mappingCone φ).X i)
    ⊢ Iff (Eq f g) (And (Eq (CategoryTheory.CategoryStruct.comp f ((↑(CochainCompl …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      A : C
      f g : Quiver.Hom A ((CochainComplex.mappingCone φ).X i)
      ⊢ Eq f g → And (Eq (CategoryTheory.CategoryStruct.comp f ((↑(CochainComplex.ma …
    -/
  · rintro rfl
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      A : C
      f : Quiver.Hom A ((CochainComplex.mappingCone φ).X i)
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp f ((↑(CochainComplex.mappingCone …
    -/
    tauto
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      A : C
      f g : Quiver.Hom A ((CochainComplex.mappingCone φ).X i)
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp f ((↑(CochainComplex.mappingCone …
    -/
  · rintro ⟨h₁, h₂⟩
    /-
      case mpr.intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      A : C
      f g : Quiver.Hom A ((CochainComplex.mappingCone φ).X i)
      h₁ : Eq (CategoryTheory.CategoryStruct.comp f ((↑(CochainComplex.mappingCone.f …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp f ((CochainComplex.mappingCone.snd …
      ⊢ Eq f g
    -/
    exact ext_to φ i j hij h₁ h₂
    /-
      🎉 no goals
    -/


lemma ext_from (i j : ℤ) (hij : j + 1 = i) {A : C} {f g : (mappingCone φ).X j ⟶ A}
                            /-
                              C : Type u_1
                              D : Type u_2
                              inst✝⁴ : CategoryTheory.Category.{?u.55899, u_1} C
                              inst✝³ : CategoryTheory.Category.{?u.55903, u_2} D
                              inst✝² : CategoryTheory.Preadditive C
                              inst✝¹ : CategoryTheory.Preadditive D
                              F G : CochainComplex C Int
                              φ : Quiver.Hom F G
                              inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                              i j : Int
                              hij : Eq (HAdd.hAdd j 1) i
                              A : C
                              f g : Quiver.Hom ((CochainComplex.mappingCone φ).X j) A
                              ⊢ Eq (HAdd.hAdd i (-1)) j
                            -/
                            /-
                              🎉 no goals
                            -/
    (h₁ : (inl φ).v i j (by omega) ≫ f = (inl φ).v i j (by omega) ≫ g)
                                                           /-
                                                             🎉 no goals
                                                           -/
    (h₂ : (inr φ).f j ≫ f = (inr φ).f j ≫ g) :
    f = g :=
  homotopyCofiber.ext_from_X φ i j hij h₁ h₂


lemma ext_from_iff (i j : ℤ) (hij : j + 1 = i) {A : C} (f g : (mappingCone φ).X j ⟶ A) :
                              /-
                                C : Type u_1
                                D : Type u_2
                                inst✝⁴ : CategoryTheory.Category.{?u.58315, u_1} C
                                inst✝³ : CategoryTheory.Category.{?u.58319, u_2} D
                                inst✝² : CategoryTheory.Preadditive C
                                inst✝¹ : CategoryTheory.Preadditive D
                                F G : CochainComplex C Int
                                φ : Quiver.Hom F G
                                inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                i j : Int
                                hij : Eq (HAdd.hAdd j 1) i
                                A : C
                                f g : Quiver.Hom ((CochainComplex.mappingCone φ).X j) A
                                ⊢ Eq (HAdd.hAdd i (-1)) j
                              -/
                              /-
                                🎉 no goals
                              -/
    f = g ↔ (inl φ).v i j (by omega) ≫ f = (inl φ).v i j (by omega) ≫ g ∧
                                                             /-
                                                               🎉 no goals
                                                             -/
      (inr φ).f j ≫ f = (inr φ).f j ≫ g := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    i j : Int
    hij : Eq (HAdd.hAdd j 1) i
    A : C
    f g : Quiver.Hom ((CochainComplex.mappingCone φ).X j) A
    ⊢ Iff (Eq f g) (And (Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.m …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd j 1) i
      A : C
      f g : Quiver.Hom ((CochainComplex.mappingCone φ).X j) A
      ⊢ Eq f g → And (Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappin …
    -/
  · rintro rfl
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd j 1) i
      A : C
      f : Quiver.Hom ((CochainComplex.mappingCone φ).X j) A
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inl …
    -/
    tauto
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd j 1) i
      A : C
      f g : Quiver.Hom ((CochainComplex.mappingCone φ).X j) A
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inl …
    -/
  · rintro ⟨h₁, h₂⟩
    /-
      case mpr.intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd j 1) i
      A : C
      f g : Quiver.Hom ((CochainComplex.mappingCone φ).X j) A
      h₁ : Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inl φ …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inr φ …
      ⊢ Eq f g
    -/
    exact ext_from φ i j hij h₁ h₂
    /-
      🎉 no goals
    -/


lemma decomp_to {i : ℤ} {A : C} (f : A ⟶ (mappingCone φ).X i) (j : ℤ) (hij : i + 1 = j) :
                                                                 /-
                                                                   C : Type u_1
                                                                   D : Type u_2
                                                                   inst✝⁴ : CategoryTheory.Category.{?u.60437, u_1} C
                                                                   inst✝³ : CategoryTheory.Category.{?u.60441, u_2} D
                                                                   inst✝² : CategoryTheory.Preadditive C
                                                                   inst✝¹ : CategoryTheory.Preadditive D
                                                                   F G : CochainComplex C Int
                                                                   φ : Quiver.Hom F G
                                                                   inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                                                   i : Int
                                                                   A : C
                                                                   f : Quiver.Hom A ((CochainComplex.mappingCone φ).X i)
                                                                   j : Int
                                                                   hij : Eq (HAdd.hAdd i 1) j
                                                                   a : Quiver.Hom A (F.X j)
                                                                   b : Quiver.Hom A (G.X i)
                                                                   ⊢ Eq (HAdd.hAdd j (-1)) i
                                                                 -/
    ∃ (a : A ⟶ F.X j) (b : A ⟶ G.X i), f = a ≫ (inl φ).v j i (by omega) + b ≫ (inr φ).f i :=
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  ⟨f ≫ (fst φ).1.v i j hij, f ≫ (snd φ).v i i (add_zero i),
       /-
         C : Type u_1
         inst✝² : CategoryTheory.Category.{u_3, u_1} C
         inst✝¹ : CategoryTheory.Preadditive C
         F G : CochainComplex C Int
         φ : Quiver.Hom F G
         inst✝ : HomologicalComplex.HasHomotopyCofiber φ
         i : Int
         A : C
         f : Quiver.Hom A ((CochainComplex.mappingCone φ).X i)
         j : Int
         hij : Eq (HAdd.hAdd i 1) j
         ⊢ Eq f (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (CategoryTheory.Category …
       -/
                                  /-
                                    🎉 no goals
                                  -/
    by apply ext_to φ i j hij <;> simp⟩
                                  /-
                                    🎉 no goals
                                  -/


lemma decomp_from {j : ℤ} {A : C} (f : (mappingCone φ).X j ⟶ A) (i : ℤ) (hij : j + 1 = i) :
    ∃ (a : F.X i ⟶ A) (b : G.X j ⟶ A),
      f = (fst φ).1.v j i hij ≫ a + (snd φ).v j j (add_zero j) ≫ b :=
                     /-
                       C : Type u_1
                       inst✝² : CategoryTheory.Category.{u_3, u_1} C
                       inst✝¹ : CategoryTheory.Preadditive C
                       F G : CochainComplex C Int
                       φ : Quiver.Hom F G
                       inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                       j : Int
                       A : C
                       f : Quiver.Hom ((CochainComplex.mappingCone φ).X j) A
                       i : Int
                       hij : Eq (HAdd.hAdd j 1) i
                       ⊢ Eq (HAdd.hAdd i (-1)) j
                     -/
  ⟨(inl φ).v i j (by omega) ≫ f, (inr φ).f j ≫ f,
                     /-
                       🎉 no goals
                     -/
       /-
         C : Type u_1
         inst✝² : CategoryTheory.Category.{u_3, u_1} C
         inst✝¹ : CategoryTheory.Preadditive C
         F G : CochainComplex C Int
         φ : Quiver.Hom F G
         inst✝ : HomologicalComplex.HasHomotopyCofiber φ
         j : Int
         A : C
         f : Quiver.Hom ((CochainComplex.mappingCone φ).X j) A
         i : Int
         hij : Eq (HAdd.hAdd j 1) i
         ⊢ Eq f (HAdd.hAdd (CategoryTheory.CategoryStruct.comp ((↑(CochainComplex.mappi …
       -/
                                    /-
                                      🎉 no goals
                                    -/
    by apply ext_from φ i j hij <;> simp⟩
                                    /-
                                      🎉 no goals
                                    -/


lemma ext_cochain_to_iff (i j : ℤ) (hij : i + 1 = j)
    {K : CochainComplex C ℤ} {γ₁ γ₂ : Cochain K (mappingCone φ) i} :
    γ₁ = γ₂ ↔ γ₁.comp (fst φ).1 hij = γ₂.comp (fst φ).1 hij ∧
      γ₁.comp (snd φ) (add_zero i) = γ₂.comp (snd φ) (add_zero i) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    i j : Int
    hij : Eq (HAdd.hAdd i 1) j
    K : CochainComplex C Int
    γ₁ γ₂ : CochainComplex.HomComplex.Cochain K (CochainComplex.mappingCone φ) i
    ⊢ Iff (Eq γ₁ γ₂) (And (Eq (γ₁.comp (↑(CochainComplex.mappingCone.fst φ)) hij)  …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ γ₂ : CochainComplex.HomComplex.Cochain K (CochainComplex.mappingCone φ) i
      ⊢ Eq γ₁ γ₂ → And (Eq (γ₁.comp (↑(CochainComplex.mappingCone.fst φ)) hij) (γ₂.c …
    -/
  · rintro rfl
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ : CochainComplex.HomComplex.Cochain K (CochainComplex.mappingCone φ) i
      ⊢ And (Eq (γ₁.comp (↑(CochainComplex.mappingCone.fst φ)) hij) (γ₁.comp (↑(Coch …
    -/
    tauto
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ γ₂ : CochainComplex.HomComplex.Cochain K (CochainComplex.mappingCone φ) i
      ⊢ And (Eq (γ₁.comp (↑(CochainComplex.mappingCone.fst φ)) hij) (γ₂.comp (↑(Coch …
    -/
  · rintro ⟨h₁, h₂⟩
    /-
      case mpr.intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ γ₂ : CochainComplex.HomComplex.Cochain K (CochainComplex.mappingCone φ) i
      h₁ : Eq (γ₁.comp (↑(CochainComplex.mappingCone.fst φ)) hij) (γ₂.comp (↑(Cochai …
      h₂ : Eq (γ₁.comp (CochainComplex.mappingCone.snd φ) ⋯) (γ₂.comp (CochainComple …
      ⊢ Eq γ₁ γ₂
    -/
    ext p q hpq
    /-
      case mpr.intro.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ γ₂ : CochainComplex.HomComplex.Cochain K (CochainComplex.mappingCone φ) i
      h₁ : Eq (γ₁.comp (↑(CochainComplex.mappingCone.fst φ)) hij) (γ₂.comp (↑(Cochai …
      h₂ : Eq (γ₁.comp (CochainComplex.mappingCone.snd φ) ⋯) (γ₂.comp (CochainComple …
      p q : Int
      hpq : Eq (HAdd.hAdd p i) q
      ⊢ Eq (γ₁.v p q hpq) (γ₂.v p q hpq)
    -/
    rw [ext_to_iff φ q (q + 1) rfl]
    /-
      case mpr.intro.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ γ₂ : CochainComplex.HomComplex.Cochain K (CochainComplex.mappingCone φ) i
      h₁ : Eq (γ₁.comp (↑(CochainComplex.mappingCone.fst φ)) hij) (γ₂.comp (↑(Cochai …
      h₂ : Eq (γ₁.comp (CochainComplex.mappingCone.snd φ) ⋯) (γ₂.comp (CochainComple …
      p q : Int
      hpq : Eq (HAdd.hAdd p i) q
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (γ₁.v p q hpq) ((↑(CochainComple …
    -/
    replace h₁ := Cochain.congr_v h₁ p (q + 1) (by omega)
    /-
      case mpr.intro.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ γ₂ : CochainComplex.HomComplex.Cochain K (CochainComplex.mappingCone φ) i
      h₂ : Eq (γ₁.comp (CochainComplex.mappingCone.snd φ) ⋯) (γ₂.comp (CochainComple …
      p q : Int
      hpq : Eq (HAdd.hAdd p i) q
      h₁ : Eq ((γ₁.comp (↑(CochainComplex.mappingCone.fst φ)) hij).v p (HAdd.hAdd q  …
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (γ₁.v p q hpq) ((↑(CochainComple …
    -/
    replace h₂ := Cochain.congr_v h₂ p q hpq
    /-
      case mpr.intro.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ γ₂ : CochainComplex.HomComplex.Cochain K (CochainComplex.mappingCone φ) i
      p q : Int
      hpq : Eq (HAdd.hAdd p i) q
      h₁ : Eq ((γ₁.comp (↑(CochainComplex.mappingCone.fst φ)) hij).v p (HAdd.hAdd q  …
      h₂ : Eq ((γ₁.comp (CochainComplex.mappingCone.snd φ) ⋯).v p q hpq) ((γ₂.comp ( …
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (γ₁.v p q hpq) ((↑(CochainComple …
    -/
    simp only [Cochain.comp_v _ _ _ p q (q + 1) hpq rfl] at h₁
    /-
      case mpr.intro.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ γ₂ : CochainComplex.HomComplex.Cochain K (CochainComplex.mappingCone φ) i
      p q : Int
      hpq : Eq (HAdd.hAdd p i) q
      h₂ : Eq ((γ₁.comp (CochainComplex.mappingCone.snd φ) ⋯).v p q hpq) ((γ₂.comp ( …
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (γ₁.v p q hpq) ((↑(CochainComplex. …
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (γ₁.v p q hpq) ((↑(CochainComple …
    -/
    simp only [Cochain.comp_zero_cochain_v] at h₂
    /-
      case mpr.intro.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ γ₂ : CochainComplex.HomComplex.Cochain K (CochainComplex.mappingCone φ) i
      p q : Int
      hpq : Eq (HAdd.hAdd p i) q
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (γ₁.v p q hpq) ((↑(CochainComplex. …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp (γ₁.v p q hpq) ((CochainComplex.ma …
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (γ₁.v p q hpq) ((↑(CochainComple …
    -/
    exact ⟨h₁, h₂⟩
    /-
      🎉 no goals
    -/


lemma ext_cochain_from_iff (i j : ℤ) (hij : i + 1 = j)
    {K : CochainComplex C ℤ} {γ₁ γ₂ : Cochain (mappingCone φ) K j} :
    γ₁ = γ₂ ↔
                                     /-
                                       C : Type u_1
                                       D : Type u_2
                                       inst✝⁴ : CategoryTheory.Category.{?u.73106, u_1} C
                                       inst✝³ : CategoryTheory.Category.{?u.73110, u_2} D
                                       inst✝² : CategoryTheory.Preadditive C
                                       inst✝¹ : CategoryTheory.Preadditive D
                                       F G : CochainComplex C Int
                                       φ : Quiver.Hom F G
                                       inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                       i j : Int
                                       hij : Eq (HAdd.hAdd i 1) j
                                       K : CochainComplex C Int
                                       γ₁ γ₂ : CochainComplex.HomComplex.Cochain (CochainComplex.mappingCone φ) K j
                                       ⊢ Eq (HAdd.hAdd (-1) j) i
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
      (inl φ).comp γ₁ (show _ = i by omega) = (inl φ).comp γ₂ (by omega) ∧
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
        (Cochain.ofHom (inr φ)).comp γ₁ (zero_add j) =
          (Cochain.ofHom (inr φ)).comp γ₂ (zero_add j) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    i j : Int
    hij : Eq (HAdd.hAdd i 1) j
    K : CochainComplex C Int
    γ₁ γ₂ : CochainComplex.HomComplex.Cochain (CochainComplex.mappingCone φ) K j
    ⊢ Iff (Eq γ₁ γ₂) (And (Eq ((CochainComplex.mappingCone.inl φ).comp γ₁ ⋯) ((Coc …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ γ₂ : CochainComplex.HomComplex.Cochain (CochainComplex.mappingCone φ) K j
      ⊢ Eq γ₁ γ₂ → And (Eq ((CochainComplex.mappingCone.inl φ).comp γ₁ ⋯) ((CochainC …
    -/
  · rintro rfl
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ : CochainComplex.HomComplex.Cochain (CochainComplex.mappingCone φ) K j
      ⊢ And (Eq ((CochainComplex.mappingCone.inl φ).comp γ₁ ⋯) ((CochainComplex.mapp …
    -/
    tauto
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ γ₂ : CochainComplex.HomComplex.Cochain (CochainComplex.mappingCone φ) K j
      ⊢ And (Eq ((CochainComplex.mappingCone.inl φ).comp γ₁ ⋯) ((CochainComplex.mapp …
    -/
  · rintro ⟨h₁, h₂⟩
    /-
      case mpr.intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ γ₂ : CochainComplex.HomComplex.Cochain (CochainComplex.mappingCone φ) K j
      h₁ : Eq ((CochainComplex.mappingCone.inl φ).comp γ₁ ⋯) ((CochainComplex.mappin …
      h₂ : Eq ((CochainComplex.HomComplex.Cochain.ofHom (CochainComplex.mappingCone. …
      ⊢ Eq γ₁ γ₂
    -/
    ext p q hpq
    /-
      case mpr.intro.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ γ₂ : CochainComplex.HomComplex.Cochain (CochainComplex.mappingCone φ) K j
      h₁ : Eq ((CochainComplex.mappingCone.inl φ).comp γ₁ ⋯) ((CochainComplex.mappin …
      h₂ : Eq ((CochainComplex.HomComplex.Cochain.ofHom (CochainComplex.mappingCone. …
      p q : Int
      hpq : Eq (HAdd.hAdd p j) q
      ⊢ Eq (γ₁.v p q hpq) (γ₂.v p q hpq)
    -/
    rw [ext_from_iff φ (p + 1) p rfl]
    /-
      case mpr.intro.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ γ₂ : CochainComplex.HomComplex.Cochain (CochainComplex.mappingCone φ) K j
      h₁ : Eq ((CochainComplex.mappingCone.inl φ).comp γ₁ ⋯) ((CochainComplex.mappin …
      h₂ : Eq ((CochainComplex.HomComplex.Cochain.ofHom (CochainComplex.mappingCone. …
      p q : Int
      hpq : Eq (HAdd.hAdd p j) q
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inl …
    -/
    replace h₁ := Cochain.congr_v h₁ (p + 1) q (by omega)
    /-
      case mpr.intro.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ γ₂ : CochainComplex.HomComplex.Cochain (CochainComplex.mappingCone φ) K j
      h₂ : Eq ((CochainComplex.HomComplex.Cochain.ofHom (CochainComplex.mappingCone. …
      p q : Int
      hpq : Eq (HAdd.hAdd p j) q
      h₁ : Eq (((CochainComplex.mappingCone.inl φ).comp γ₁ ⋯).v (HAdd.hAdd p 1) q ⋯) …
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inl …
    -/
    replace h₂ := Cochain.congr_v h₂ p q (by omega)
    /-
      case mpr.intro.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ γ₂ : CochainComplex.HomComplex.Cochain (CochainComplex.mappingCone φ) K j
      p q : Int
      hpq : Eq (HAdd.hAdd p j) q
      h₁ : Eq (((CochainComplex.mappingCone.inl φ).comp γ₁ ⋯).v (HAdd.hAdd p 1) q ⋯) …
      h₂ : Eq (((CochainComplex.HomComplex.Cochain.ofHom (CochainComplex.mappingCone …
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inl …
    -/
    simp only [Cochain.comp_v (inl φ) _ _ (p + 1) p q (by omega) hpq] at h₁
    /-
      case mpr.intro.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ γ₂ : CochainComplex.HomComplex.Cochain (CochainComplex.mappingCone φ) K j
      p q : Int
      hpq : Eq (HAdd.hAdd p j) q
      h₂ : Eq (((CochainComplex.HomComplex.Cochain.ofHom (CochainComplex.mappingCone …
      h₁ : Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inl φ …
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inl …
    -/
    simp only [Cochain.zero_cochain_comp_v, Cochain.ofHom_v] at h₂
    /-
      case mpr.intro.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      i j : Int
      hij : Eq (HAdd.hAdd i 1) j
      K : CochainComplex C Int
      γ₁ γ₂ : CochainComplex.HomComplex.Cochain (CochainComplex.mappingCone φ) K j
      p q : Int
      hpq : Eq (HAdd.hAdd p j) q
      h₁ : Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inl φ …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inr φ …
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inl …
    -/
    exact ⟨h₁, h₂⟩
    /-
      🎉 no goals
    -/


lemma id :
    (fst φ).1.comp (inl φ) (add_neg_cancel 1) +
      (snd φ).comp (Cochain.ofHom (inr φ)) (add_zero 0) = Cochain.ofHom (𝟙 _) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    ⊢ Eq (HAdd.hAdd ((↑(CochainComplex.mappingCone.fst φ)).comp (CochainComplex.ma …
  -/
  simp [ext_cochain_from_iff φ (-1) 0 (neg_add_cancel 1)]
  /-
    🎉 no goals
  -/


lemma id_X (p q : ℤ) (hpq : p + 1 = q) :
                                            /-
                                              C : Type u_1
                                              D : Type u_2
                                              inst✝⁴ : CategoryTheory.Category.{?u.81545, u_1} C
                                              inst✝³ : CategoryTheory.Category.{?u.81549, u_2} D
                                              inst✝² : CategoryTheory.Preadditive C
                                              inst✝¹ : CategoryTheory.Preadditive D
                                              F G : CochainComplex C Int
                                              φ : Quiver.Hom F G
                                              inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                              p q : Int
                                              hpq : Eq (HAdd.hAdd p 1) q
                                              ⊢ Eq (HAdd.hAdd q (-1)) p
                                            -/
    (fst φ).1.v p q hpq ≫ (inl φ).v q p (by omega) +
                                            /-
                                              🎉 no goals
                                            -/
      (snd φ).v p p (add_zero p) ≫ (inr φ).f p = 𝟙 ((mappingCone φ).X p) := by
  simpa only [Cochain.add_v, Cochain.comp_zero_cochain_v, Cochain.ofHom_v, id_f,
    Cochain.comp_v _ _ (add_neg_cancel 1) p q p hpq (by omega)]
    using Cochain.congr_v (id φ) p p (add_zero p)


@[reassoc]
lemma inl_v_d (i j k : ℤ) (hij : i + (-1) = j) (hik : k + (-1) = i) :
    (inl φ).v i j hij ≫ (mappingCone φ).d j i =
      φ.f i ≫ (inr φ).f i - F.d i k ≫ (inl φ).v _ _ hik := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    i j k : Int
    hij : Eq (HAdd.hAdd i (-1)) j
    hik : Eq (HAdd.hAdd k (-1)) i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inl φ).v …
  -/
  dsimp [mappingCone, inl, inr]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    i j k : Int
    hij : Eq (HAdd.hAdd i (-1)) j
    hik : Eq (HAdd.hAdd k (-1)) i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
  -/
  rw [homotopyCofiber.inlX_d φ j i k (by dsimp; omega) (by dsimp; omega)]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    i j k : Int
    hij : Eq (HAdd.hAdd i (-1)) j
    hik : Eq (HAdd.hAdd k (-1)) i
    ⊢ Eq (HAdd.hAdd (Neg.neg (CategoryTheory.CategoryStruct.comp (F.d i k) (Homolo …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp 1100)]
lemma inr_f_d (n₁ n₂ : ℤ) :
    (inr φ).f n₁ ≫ (mappingCone φ).d n₁ n₂ = G.d n₁ n₂ ≫ (inr φ).f n₂ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    n₁ n₂ : Int
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inr φ).f …
  -/
  apply Hom.comm
  /-
    🎉 no goals
  -/


@[reassoc]
lemma d_fst_v (i j k : ℤ) (hij : i + 1 = j) (hjk : j + 1 = k) :
    (mappingCone φ).d i j ≫ (fst φ).1.v j k hjk =
      -(fst φ).1.v i j hij ≫ F.d j k := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    i j k : Int
    hij : Eq (HAdd.hAdd i 1) j
    hjk : Eq (HAdd.hAdd j 1) k
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone φ).d i j …
  -/
  apply homotopyCofiber.d_fstX
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma d_fst_v' (i j : ℤ) (hij : i + 1 = j) :
    (mappingCone φ).d (i - 1) i ≫ (fst φ).1.v i j hij =
                                 /-
                                   C : Type u_1
                                   D : Type u_2
                                   inst✝⁴ : CategoryTheory.Category.{?u.94431, u_1} C
                                   inst✝³ : CategoryTheory.Category.{?u.94435, u_2} D
                                   inst✝² : CategoryTheory.Preadditive C
                                   inst✝¹ : CategoryTheory.Preadditive D
                                   F G : CochainComplex C Int
                                   φ : Quiver.Hom F G
                                   inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                   i j : Int
                                   hij : Eq (HAdd.hAdd i 1) j
                                   ⊢ Eq (HAdd.hAdd (HSub.hSub i 1) 1) i
                                 -/
      -(fst φ).1.v (i - 1) i (by omega) ≫ F.d i j :=
                                 /-
                                   🎉 no goals
                                 -/
                            /-
                              C : Type u_1
                              inst✝² : CategoryTheory.Category.{u_3, u_1} C
                              inst✝¹ : CategoryTheory.Preadditive C
                              F G : CochainComplex C Int
                              φ : Quiver.Hom F G
                              inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                              i j : Int
                              hij : Eq (HAdd.hAdd i 1) j
                              ⊢ Eq (HAdd.hAdd (HSub.hSub i 1) 1) i
                            -/
  d_fst_v φ (i - 1) i j (by omega) hij
                            /-
                              🎉 no goals
                            -/


@[reassoc]
lemma d_snd_v (i j : ℤ) (hij : i + 1 = j) :
    (mappingCone φ).d i j ≫ (snd φ).v j j (add_zero _) =
      (fst φ).1.v i j hij ≫ φ.f j + (snd φ).v i i (add_zero i) ≫ G.d i j := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    i j : Int
    hij : Eq (HAdd.hAdd i 1) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone φ).d i j …
  -/
  dsimp [mappingCone, snd, fst]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    i j : Int
    hij : Eq (HAdd.hAdd i 1) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.d …
  -/
  simp only [Cochain.ofHoms_v]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    i j : Int
    hij : Eq (HAdd.hAdd i 1) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.d …
  -/
  apply homotopyCofiber.d_sndX
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma d_snd_v' (n : ℤ) :
    (mappingCone φ).d (n - 1) n ≫ (snd φ).v n n (add_zero n) =
                                                          /-
                                                            C : Type u_1
                                                            D : Type u_2
                                                            inst✝⁴ : CategoryTheory.Category.{?u.100115, u_1} C
                                                            inst✝³ : CategoryTheory.Category.{?u.100119, u_2} D
                                                            inst✝² : CategoryTheory.Preadditive C
                                                            inst✝¹ : CategoryTheory.Preadditive D
                                                            F G : CochainComplex C Int
                                                            φ : Quiver.Hom F G
                                                            inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                                            n : Int
                                                            ⊢ Eq (HAdd.hAdd (HSub.hSub n 1) 1) n
                                                          -/
    (fst φ : Cochain (mappingCone φ) F 1).v (n - 1) n (by omega) ≫ φ.f n +
                                                          /-
                                                            🎉 no goals
                                                          -/
      (snd φ).v (n - 1) (n - 1) (add_zero _) ≫ G.d (n - 1) n := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    n : Int
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone φ).d (HS …
  -/
  apply d_snd_v
  /-
    🎉 no goals
  -/


@[simp]
lemma δ_inl :
    δ (-1) 0 (inl φ) = Cochain.ofHom (φ ≫ inr φ) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    ⊢ Eq (CochainComplex.HomComplex.δ (-1) 0 (CochainComplex.mappingCone.inl φ)) ( …
  -/
  ext p
  simp [δ_v (-1) 0 (neg_add_cancel 1) (inl φ) p p (add_zero p) _ _ rfl rfl,
    inl_v_d φ p (p - 1) (p + 1) (by omega) (by omega)]


@[simp]
lemma δ_snd :
    δ 0 1 (snd φ) = -(fst φ).1.comp (Cochain.ofHom φ) (add_zero 1) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    ⊢ Eq (CochainComplex.HomComplex.δ 0 1 (CochainComplex.mappingCone.snd φ)) (Neg …
  -/
  ext p q hpq
  /-
    case h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    p q : Int
    hpq : Eq (HAdd.hAdd p 1) q
    ⊢ Eq ((CochainComplex.HomComplex.δ 0 1 (CochainComplex.mappingCone.snd φ)).v p …
  -/
  simp [d_snd_v φ p q hpq]
  /-
    🎉 no goals
  -/


/-- Given `φ : F ⟶ G`, this is the cochain in `Cochain (mappingCone φ) K n` that is
constructed from two cochains `α : Cochain F K m` (with `m + 1 = n`) and `β : Cochain F K n`. -/
noncomputable def descCochain (α : Cochain F K m) (β : Cochain G K n) (h : m + 1 = n) :
    Cochain (mappingCone φ) K n :=
                       /-
                         C : Type u_1
                         D : Type u_2
                         inst✝⁴ : CategoryTheory.Category.{?u.109262, u_1} C
                         inst✝³ : CategoryTheory.Category.{?u.109266, u_2} D
                         inst✝² : CategoryTheory.Preadditive C
                         inst✝¹ : CategoryTheory.Preadditive D
                         F G : CochainComplex C Int
                         φ : Quiver.Hom F G
                         inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                         K : CochainComplex C Int
                         n m : Int
                         α : CochainComplex.HomComplex.Cochain F K m
                         β : CochainComplex.HomComplex.Cochain G K n
                         h : Eq (HAdd.hAdd m 1) n
                         ⊢ Eq (HAdd.hAdd 1 m) n
                       -/
  (fst φ).1.comp α (by rw [← h, add_comm]) + (snd φ).comp β (zero_add n)
                       /-
                         🎉 no goals
                       -/


@[simp]
lemma inl_descCochain :
                                           /-
                                             C : Type u_1
                                             D : Type u_2
                                             inst✝⁴ : CategoryTheory.Category.{?u.112606, u_1} C
                                             inst✝³ : CategoryTheory.Category.{?u.112610, u_2} D
                                             inst✝² : CategoryTheory.Preadditive C
                                             inst✝¹ : CategoryTheory.Preadditive D
                                             F G : CochainComplex C Int
                                             φ : Quiver.Hom F G
                                             inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                             K : CochainComplex C Int
                                             n m : Int
                                             α : CochainComplex.HomComplex.Cochain F K m
                                             β : CochainComplex.HomComplex.Cochain G K n
                                             h : Eq (HAdd.hAdd m 1) n
                                             ⊢ Eq (HAdd.hAdd (-1) n) m
                                           -/
    (inl φ).comp (descCochain φ α β h) (by omega) = α := by
                                           /-
                                             🎉 no goals
                                           -/
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    n m : Int
    α : CochainComplex.HomComplex.Cochain F K m
    β : CochainComplex.HomComplex.Cochain G K n
    h : Eq (HAdd.hAdd m 1) n
    ⊢ Eq ((CochainComplex.mappingCone.inl φ).comp (CochainComplex.mappingCone.desc …
  -/
  simp [descCochain]
  /-
    🎉 no goals
  -/


@[simp]
lemma inr_descCochain :
    (Cochain.ofHom (inr φ)).comp (descCochain φ α β h) (zero_add n) = β := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    n m : Int
    α : CochainComplex.HomComplex.Cochain F K m
    β : CochainComplex.HomComplex.Cochain G K n
    h : Eq (HAdd.hAdd m 1) n
    ⊢ Eq ((CochainComplex.HomComplex.Cochain.ofHom (CochainComplex.mappingCone.inr …
  -/
  simp [descCochain]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inl_v_descCochain_v (p₁ p₂ p₃ : ℤ) (h₁₂ : p₁ + (-1) = p₂) (h₂₃ : p₂ + n = p₃) :
    (inl φ).v p₁ p₂ h₁₂ ≫ (descCochain φ α β h).v p₂ p₃ h₂₃ =
                      /-
                        C : Type u_1
                        D : Type u_2
                        inst✝⁴ : CategoryTheory.Category.{?u.116736, u_1} C
                        inst✝³ : CategoryTheory.Category.{?u.116740, u_2} D
                        inst✝² : CategoryTheory.Preadditive C
                        inst✝¹ : CategoryTheory.Preadditive D
                        F G : CochainComplex C Int
                        φ : Quiver.Hom F G
                        inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                        K : CochainComplex C Int
                        n m : Int
                        α : CochainComplex.HomComplex.Cochain F K m
                        β : CochainComplex.HomComplex.Cochain G K n
                        h : Eq (HAdd.hAdd m 1) n
                        p₁ p₂ p₃ : Int
                        h₁₂ : Eq (HAdd.hAdd p₁ (-1)) p₂
                        h₂₃ : Eq (HAdd.hAdd p₂ n) p₃
                        ⊢ Eq (HAdd.hAdd p₁ m) p₃
                      -/
        α.v p₁ p₃ (by rw [← h₂₃, ← h₁₂, ← h, add_comm m, add_assoc, neg_add_cancel_left]) := by
                      /-
                        🎉 no goals
                      -/
  simpa only [Cochain.comp_v _ _ (show -1 + n = m by omega) p₁ p₂ p₃
    (by omega) (by omega)] using
      Cochain.congr_v (inl_descCochain φ α β h) p₁ p₃ (by omega)


@[reassoc (attr := simp)]
lemma inr_f_descCochain_v (p₁ p₂ : ℤ) (h₁₂ : p₁ + n = p₂) :
    (inr φ).f p₁ ≫ (descCochain φ α β h).v p₁ p₂ h₁₂ = β.v p₁ p₂ h₁₂ := by
  simpa only [Cochain.comp_v _ _ (zero_add n) p₁ p₁ p₂ (add_zero p₁) h₁₂, Cochain.ofHom_v]
    using Cochain.congr_v (inr_descCochain φ α β h) p₁ p₂ (by omega)


lemma δ_descCochain (n' : ℤ) (hn' : n + 1 = n') :
    δ n n' (descCochain φ α β h) =
      (fst φ).1.comp (δ m n α +
                                                                    /-
                                                                      C : Type u_1
                                                                      D : Type u_2
                                                                      inst✝⁴ : CategoryTheory.Category.{?u.125832, u_1} C
                                                                      inst✝³ : CategoryTheory.Category.{?u.125836, u_2} D
                                                                      inst✝² : CategoryTheory.Preadditive C
                                                                      inst✝¹ : CategoryTheory.Preadditive D
                                                                      F G : CochainComplex C Int
                                                                      φ : Quiver.Hom F G
                                                                      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                                                      K : CochainComplex C Int
                                                                      n m : Int
                                                                      α : CochainComplex.HomComplex.Cochain F K m
                                                                      β : CochainComplex.HomComplex.Cochain G K n
                                                                      h : Eq (HAdd.hAdd m 1) n
                                                                      n' : Int
                                                                      hn' : Eq (HAdd.hAdd n 1) n'
                                                                      ⊢ Eq (HAdd.hAdd 1 n) n'
                                                                    -/
          n'.negOnePow • (Cochain.ofHom φ).comp β (zero_add n)) (by omega) +
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
      (snd φ).comp (δ n n' β) (zero_add n') := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    n m : Int
    α : CochainComplex.HomComplex.Cochain F K m
    β : CochainComplex.HomComplex.Cochain G K n
    h : Eq (HAdd.hAdd m 1) n
    n' : Int
    hn' : Eq (HAdd.hAdd n 1) n'
    ⊢ Eq (CochainComplex.HomComplex.δ n n' (CochainComplex.mappingCone.descCochain …
  -/
  dsimp only [descCochain]
  simp only [δ_add, Cochain.comp_add, δ_comp (fst φ).1 α _ 2 n n' hn' (by omega) (by omega),
    Cocycle.δ_eq_zero, Cochain.zero_comp, smul_zero, add_zero,
    δ_comp (snd φ) β (zero_add n) 1 n' n' hn' (zero_add 1) hn', δ_snd, Cochain.neg_comp,
    smul_neg, Cochain.comp_assoc_of_second_is_zero_cochain, Cochain.comp_units_smul, ← hn',
    Int.negOnePow_succ, Units.neg_smul, Cochain.comp_neg]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    n m : Int
    α : CochainComplex.HomComplex.Cochain F K m
    β : CochainComplex.HomComplex.Cochain G K n
    h : Eq (HAdd.hAdd m 1) n
    n' : Int
    hn' : Eq (HAdd.hAdd n 1) n'
    ⊢ Eq (HAdd.hAdd ((↑(CochainComplex.mappingCone.fst φ)).comp (CochainComplex.Ho …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


/-- Given `φ : F ⟶ G`, this is the cocycle in `Cocycle (mappingCone φ) K n` that is
constructed from `α : Cochain F K m` (with `m + 1 = n`) and `β : Cocycle F K n`,
when a suitable cocycle relation is satisfied. -/
@[simps!]
noncomputable def descCocycle {K : CochainComplex C ℤ} {n m : ℤ}
    (α : Cochain F K m) (β : Cocycle G K n)
    (h : m + 1 = n) (eq : δ m n α = n.negOnePow • (Cochain.ofHom φ).comp β.1 (zero_add n)) :
    Cocycle (mappingCone φ) K n :=
  Cocycle.mk (descCochain φ α β.1 h) (n + 1) rfl
        /-
          C : Type u_1
          D : Type u_2
          inst✝⁴ : CategoryTheory.Category.{?u.132753, u_1} C
          inst✝³ : CategoryTheory.Category.{?u.132757, u_2} D
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.Preadditive D
          F G : CochainComplex C Int
          φ : Quiver.Hom F G
          inst✝ : HomologicalComplex.HasHomotopyCofiber φ
          K : CochainComplex C Int
          n m : Int
          α : CochainComplex.HomComplex.Cochain F K m
          β : CochainComplex.HomComplex.Cocycle G K n
          h : Eq (HAdd.hAdd m 1) n
          eq : Eq (CochainComplex.HomComplex.δ m n α) (HSMul.hSMul n.negOnePow ((Cochain …
          ⊢ Eq (CochainComplex.HomComplex.δ n (HAdd.hAdd n 1) (CochainComplex.mappingCon …
        -/
    (by simp [δ_descCochain _ _ _ _ _ rfl, eq, Int.negOnePow_succ])
        /-
          🎉 no goals
        -/


/-- Given `φ : F ⟶ G`, this is the morphism `mappingCone φ ⟶ K` that is constructed
from a cochain `α : Cochain F K (-1)` and a morphism `β : G ⟶ K` such that
`δ (-1) 0 α = Cochain.ofHom (φ ≫ β)`. -/
noncomputable def desc (α : Cochain F K (-1)) (β : G ⟶ K)
    (eq : δ (-1) 0 α = Cochain.ofHom (φ ≫ β)) : mappingCone φ ⟶ K :=
                                                                          /-
                                                                            C : Type u_1
                                                                            D : Type u_2
                                                                            inst✝⁴ : CategoryTheory.Category.{?u.138335, u_1} C
                                                                            inst✝³ : CategoryTheory.Category.{?u.138339, u_2} D
                                                                            inst✝² : CategoryTheory.Preadditive C
                                                                            inst✝¹ : CategoryTheory.Preadditive D
                                                                            F G : CochainComplex C Int
                                                                            φ : Quiver.Hom F G
                                                                            inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                                                            K : CochainComplex C Int
                                                                            α : CochainComplex.HomComplex.Cochain F K (-1)
                                                                            β : Quiver.Hom G K
                                                                            eq : Eq (CochainComplex.HomComplex.δ (-1) 0 α) (CochainComplex.HomComplex.Coch …
                                                                            ⊢ Eq (CochainComplex.HomComplex.δ (-1) 0 α) (HSMul.hSMul (Int.negOnePow 0) ((C …
                                                                          -/
  Cocycle.homOf (descCocycle φ α (Cocycle.ofHom β) (neg_add_cancel 1) (by simp [eq]))
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
lemma ofHom_desc :
    Cochain.ofHom (desc φ α β eq) = descCochain φ α (Cochain.ofHom β) (neg_add_cancel 1) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    α : CochainComplex.HomComplex.Cochain F K (-1)
    β : Quiver.Hom G K
    eq : Eq (CochainComplex.HomComplex.δ (-1) 0 α) (CochainComplex.HomComplex.Coch …
    ⊢ Eq (CochainComplex.HomComplex.Cochain.ofHom (CochainComplex.mappingCone.desc …
  -/
  simp [desc]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inl_v_desc_f (p q : ℤ) (h : p + (-1) = q) :
    (inl φ).v p q h ≫ (desc φ α β eq).f q = α.v p q h := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    α : CochainComplex.HomComplex.Cochain F K (-1)
    β : Quiver.Hom G K
    eq : Eq (CochainComplex.HomComplex.δ (-1) 0 α) (CochainComplex.HomComplex.Coch …
    p q : Int
    h : Eq (HAdd.hAdd p (-1)) q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inl φ).v …
  -/
  simp [desc]
  /-
    🎉 no goals
  -/


lemma inl_desc :
    (inl φ).comp (Cochain.ofHom (desc φ α β eq)) (add_zero _) = α := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    α : CochainComplex.HomComplex.Cochain F K (-1)
    β : Quiver.Hom G K
    eq : Eq (CochainComplex.HomComplex.δ (-1) 0 α) (CochainComplex.HomComplex.Coch …
    ⊢ Eq ((CochainComplex.mappingCone.inl φ).comp (CochainComplex.HomComplex.Cocha …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inr_f_desc_f (p : ℤ) :
    (inr φ).f p ≫ (desc φ α β eq).f p = β.f p := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    α : CochainComplex.HomComplex.Cochain F K (-1)
    β : Quiver.Hom G K
    eq : Eq (CochainComplex.HomComplex.δ (-1) 0 α) (CochainComplex.HomComplex.Coch …
    p : Int
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inr φ).f …
  -/
  simp [desc]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
                                                 /-
                                                   C : Type u_1
                                                   inst✝² : CategoryTheory.Category.{u_3, u_1} C
                                                   inst✝¹ : CategoryTheory.Preadditive C
                                                   F G : CochainComplex C Int
                                                   φ : Quiver.Hom F G
                                                   inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                                   K : CochainComplex C Int
                                                   α : CochainComplex.HomComplex.Cochain F K (-1)
                                                   β : Quiver.Hom G K
                                                   eq : Eq (CochainComplex.HomComplex.δ (-1) 0 α) (CochainComplex.HomComplex.Coch …
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CochainComplex.mappingCone.inr φ) (C …
                                                 -/
lemma inr_desc : inr φ ≫ desc φ α β eq = β := by aesop_cat
                                                 /-
                                                   🎉 no goals
                                                 -/


lemma desc_f (p q : ℤ) (hpq : p + 1 = q) :
                                                            /-
                                                              C : Type u_1
                                                              D : Type u_2
                                                              inst✝⁴ : CategoryTheory.Category.{?u.155372, u_1} C
                                                              inst✝³ : CategoryTheory.Category.{?u.155376, u_2} D
                                                              inst✝² : CategoryTheory.Preadditive C
                                                              inst✝¹ : CategoryTheory.Preadditive D
                                                              F G : CochainComplex C Int
                                                              φ : Quiver.Hom F G
                                                              inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                                              K : CochainComplex C Int
                                                              α : CochainComplex.HomComplex.Cochain F K (-1)
                                                              β : Quiver.Hom G K
                                                              eq : Eq (CochainComplex.HomComplex.δ (-1) 0 α) (CochainComplex.HomComplex.Coch …
                                                              p q : Int
                                                              hpq : Eq (HAdd.hAdd p 1) q
                                                              ⊢ Eq (HAdd.hAdd q (-1)) p
                                                            -/
    (desc φ α β eq).f p = (fst φ).1.v p q hpq ≫ α.v q p (by omega) +
                                                            /-
                                                              🎉 no goals
                                                            -/
      (snd φ).v p p (add_zero p) ≫ β.f p := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    α : CochainComplex.HomComplex.Cochain F K (-1)
    β : Quiver.Hom G K
    eq : Eq (CochainComplex.HomComplex.δ (-1) 0 α) (CochainComplex.HomComplex.Coch …
    p q : Int
    hpq : Eq (HAdd.hAdd p 1) q
    ⊢ Eq ((CochainComplex.mappingCone.desc φ α β eq).f p) (HAdd.hAdd (CategoryTheo …
  -/
  simp [ext_from_iff _ _ _ hpq]
  /-
    🎉 no goals
  -/


/-- Constructor for homotopies between morphisms from a mapping cone. -/
noncomputable def descHomotopy {K : CochainComplex C ℤ} (f₁ f₂ : mappingCone φ ⟶ K)
    (γ₁ : Cochain F K (-2)) (γ₂ : Cochain G K (-1))
    (h₁ : (inl φ).comp (Cochain.ofHom f₁) (add_zero (-1))  =
      δ (-2) (-1) γ₁ + (Cochain.ofHom φ).comp γ₂ (zero_add (-1)) +
      (inl φ).comp (Cochain.ofHom f₂) (add_zero (-1)))
    (h₂ : Cochain.ofHom (inr φ ≫ f₁) = δ (-1) 0 γ₂ + Cochain.ofHom (inr φ ≫ f₂)) :
    Homotopy f₁ f₂ :=
                                                              /-
                                                                C : Type u_1
                                                                D : Type u_2
                                                                inst✝⁴ : CategoryTheory.Category.{?u.160506, u_1} C
                                                                inst✝³ : CategoryTheory.Category.{?u.160510, u_2} D
                                                                inst✝² : CategoryTheory.Preadditive C
                                                                inst✝¹ : CategoryTheory.Preadditive D
                                                                F G : CochainComplex C Int
                                                                φ : Quiver.Hom F G
                                                                inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                                                K : CochainComplex C Int
                                                                f₁ f₂ : Quiver.Hom (CochainComplex.mappingCone φ) K
                                                                γ₁ : CochainComplex.HomComplex.Cochain F K (-2)
                                                                γ₂ : CochainComplex.HomComplex.Cochain G K (-1)
                                                                h₁ : Eq ((CochainComplex.mappingCone.inl φ).comp (CochainComplex.HomComplex.Co …
                                                                h₂ : Eq (CochainComplex.HomComplex.Cochain.ofHom (CategoryTheory.CategoryStruc …
                                                                ⊢ Eq (HAdd.hAdd (-2) 1) (-1)
                                                              -/
  (Cochain.equivHomotopy f₁ f₂).symm ⟨descCochain φ γ₁ γ₂ (by norm_num), by
                                                              /-
                                                                🎉 no goals
                                                              -/
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.160506, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.160510, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Preadditive D
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      K : CochainComplex C Int
      f₁ f₂ : Quiver.Hom (CochainComplex.mappingCone φ) K
      γ₁ : CochainComplex.HomComplex.Cochain F K (-2)
      γ₂ : CochainComplex.HomComplex.Cochain G K (-1)
      h₁ : Eq ((CochainComplex.mappingCone.inl φ).comp (CochainComplex.HomComplex.Co …
      h₂ : Eq (CochainComplex.HomComplex.Cochain.ofHom (CategoryTheory.CategoryStruc …
      ⊢ Eq (CochainComplex.HomComplex.Cochain.ofHom f₁) (HAdd.hAdd (CochainComplex.H …
    -/
    simp only [Cochain.ofHom_comp] at h₂
    simp [ext_cochain_from_iff _ _ _ (neg_add_cancel 1),
      δ_descCochain _ _ _ _ _ (neg_add_cancel 1), h₁, h₂]⟩


/-- Given `φ : F ⟶ G`, this is the cochain in `Cochain (mappingCone φ) K n` that is
constructed from two cochains `α : Cochain F K m` (with `m + 1 = n`) and `β : Cochain F K n`. -/
noncomputable def liftCochain (α : Cochain K F m) (β : Cochain K G n) (h : n + 1 = m) :
    Cochain K (mappingCone φ) n :=
                     /-
                       C : Type u_1
                       D : Type u_2
                       inst✝⁴ : CategoryTheory.Category.{?u.172073, u_1} C
                       inst✝³ : CategoryTheory.Category.{?u.172077, u_2} D
                       inst✝² : CategoryTheory.Preadditive C
                       inst✝¹ : CategoryTheory.Preadditive D
                       F G : CochainComplex C Int
                       φ : Quiver.Hom F G
                       inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                       K : CochainComplex C Int
                       n m : Int
                       α : CochainComplex.HomComplex.Cochain K F m
                       β : CochainComplex.HomComplex.Cochain K G n
                       h : Eq (HAdd.hAdd n 1) m
                       ⊢ Eq (HAdd.hAdd m (-1)) n
                     -/
  α.comp (inl φ) (by omega) + β.comp (Cochain.ofHom (inr φ)) (add_zero n)
                     /-
                       🎉 no goals
                     -/


@[simp]
lemma liftCochain_fst :
    (liftCochain φ α β h).comp (fst φ).1 h = α := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    n m : Int
    α : CochainComplex.HomComplex.Cochain K F m
    β : CochainComplex.HomComplex.Cochain K G n
    h : Eq (HAdd.hAdd n 1) m
    ⊢ Eq ((CochainComplex.mappingCone.liftCochain φ α β h).comp (↑(CochainComplex. …
  -/
  simp [liftCochain]
  /-
    🎉 no goals
  -/


@[simp]
lemma liftCochain_snd :
    (liftCochain φ α β h).comp (snd φ) (add_zero n) = β := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    n m : Int
    α : CochainComplex.HomComplex.Cochain K F m
    β : CochainComplex.HomComplex.Cochain K G n
    h : Eq (HAdd.hAdd n 1) m
    ⊢ Eq ((CochainComplex.mappingCone.liftCochain φ α β h).comp (CochainComplex.ma …
  -/
  simp [liftCochain]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma liftCochain_v_fst_v (p₁ p₂ p₃ : ℤ) (h₁₂ : p₁ + n = p₂) (h₂₃ : p₂ + 1 = p₃) :
                                                                              /-
                                                                                C : Type u_1
                                                                                D : Type u_2
                                                                                inst✝⁴ : CategoryTheory.Category.{?u.179987, u_1} C
                                                                                inst✝³ : CategoryTheory.Category.{?u.179991, u_2} D
                                                                                inst✝² : CategoryTheory.Preadditive C
                                                                                inst✝¹ : CategoryTheory.Preadditive D
                                                                                F G : CochainComplex C Int
                                                                                φ : Quiver.Hom F G
                                                                                inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                                                                K : CochainComplex C Int
                                                                                n m : Int
                                                                                α : CochainComplex.HomComplex.Cochain K F m
                                                                                β : CochainComplex.HomComplex.Cochain K G n
                                                                                h : Eq (HAdd.hAdd n 1) m
                                                                                p₁ p₂ p₃ : Int
                                                                                h₁₂ : Eq (HAdd.hAdd p₁ n) p₂
                                                                                h₂₃ : Eq (HAdd.hAdd p₂ 1) p₃
                                                                                ⊢ Eq (HAdd.hAdd p₁ m) p₃
                                                                              -/
    (liftCochain φ α β h).v p₁ p₂ h₁₂ ≫ (fst φ).1.v p₂ p₃ h₂₃ = α.v p₁ p₃ (by omega) := by
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
  simpa only [Cochain.comp_v _ _ h p₁ p₂ p₃ h₁₂ h₂₃]
    using Cochain.congr_v (liftCochain_fst φ α β h) p₁ p₃ (by omega)


@[reassoc (attr := simp)]
lemma liftCochain_v_snd_v (p₁ p₂ : ℤ) (h₁₂ : p₁ + n = p₂) :
    (liftCochain φ α β h).v p₁ p₂ h₁₂ ≫ (snd φ).v p₂ p₂ (add_zero p₂) = β.v p₁ p₂ h₁₂ := by
  simpa only [Cochain.comp_v _ _ (add_zero n) p₁ p₂ p₂ h₁₂ (add_zero p₂)]
    using Cochain.congr_v (liftCochain_snd φ α β h) p₁ p₂ (by omega)


lemma δ_liftCochain (m' : ℤ) (hm' : m + 1 = m') :
                                                               /-
                                                                 C : Type u_1
                                                                 D : Type u_2
                                                                 inst✝⁴ : CategoryTheory.Category.{?u.187752, u_1} C
                                                                 inst✝³ : CategoryTheory.Category.{?u.187756, u_2} D
                                                                 inst✝² : CategoryTheory.Preadditive C
                                                                 inst✝¹ : CategoryTheory.Preadditive D
                                                                 F G : CochainComplex C Int
                                                                 φ : Quiver.Hom F G
                                                                 inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                                                 K : CochainComplex C Int
                                                                 n m : Int
                                                                 α : CochainComplex.HomComplex.Cochain K F m
                                                                 β : CochainComplex.HomComplex.Cochain K G n
                                                                 h : Eq (HAdd.hAdd n 1) m
                                                                 m' : Int
                                                                 hm' : Eq (HAdd.hAdd m 1) m'
                                                                 ⊢ Eq (HAdd.hAdd m' (-1)) m
                                                               -/
    δ n m (liftCochain φ α β h) = -(δ m m' α).comp (inl φ) (by omega) +
                                                               /-
                                                                 🎉 no goals
                                                               -/
      (δ n m β + α.comp (Cochain.ofHom φ) (add_zero m)).comp
        (Cochain.ofHom (inr φ)) (add_zero m) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    n m : Int
    α : CochainComplex.HomComplex.Cochain K F m
    β : CochainComplex.HomComplex.Cochain K G n
    h : Eq (HAdd.hAdd n 1) m
    m' : Int
    hm' : Eq (HAdd.hAdd m 1) m'
    ⊢ Eq (CochainComplex.HomComplex.δ n m (CochainComplex.mappingCone.liftCochain  …
  -/
  dsimp only [liftCochain]
  simp only [δ_add, δ_comp α (inl φ) _ m' _ _ h hm' (neg_add_cancel 1),
    δ_comp_zero_cochain _ _ _ h, δ_inl, Cochain.ofHom_comp,
    Int.negOnePow_neg, Int.negOnePow_one, Units.neg_smul, one_smul,
    δ_ofHom, Cochain.comp_zero, zero_add, Cochain.add_comp,
    Cochain.comp_assoc_of_second_is_zero_cochain]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    n m : Int
    α : CochainComplex.HomComplex.Cochain K F m
    β : CochainComplex.HomComplex.Cochain K G n
    h : Eq (HAdd.hAdd n 1) m
    m' : Int
    hm' : Eq (HAdd.hAdd m 1) m'
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (α.comp ((CochainComplex.HomComplex.Cochain.ofHom φ …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


/-- Given `φ : F ⟶ G`, this is the cocycle in `Cocycle K (mappingCone φ) n` that is
constructed from `α : Cochain K F m` (with `n + 1 = m`) and `β : Cocycle K G n`,
when a suitable cocycle relation is satisfied. -/
@[simps!]
noncomputable def liftCocycle {K : CochainComplex C ℤ} {n m : ℤ}
    (α : Cocycle K F m) (β : Cochain K G n) (h : n + 1 = m)
    (eq : δ n m β + α.1.comp (Cochain.ofHom φ) (add_zero m) = 0) :
    Cocycle K (mappingCone φ) n :=
  Cocycle.mk (liftCochain φ α β h) m h (by
    simp only [δ_liftCochain φ α β h (m+1) rfl, eq,
      Cocycle.δ_eq_zero, Cochain.zero_comp, neg_zero, add_zero])


/-- Given `φ : F ⟶ G`, this is the morphism `K ⟶ mappingCone φ` that is constructed
from a cocycle `α : Cochain K F 1` and a cochain `β : Cochain K G 0`
when a suitable cocycle relation is satisfied. -/
noncomputable def lift :
    K ⟶ mappingCone φ :=
  Cocycle.homOf (liftCocycle φ α β (zero_add 1) eq)


@[simp]
lemma ofHom_lift :
    Cochain.ofHom (lift φ α β eq) = liftCochain φ α β (zero_add 1) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    α : CochainComplex.HomComplex.Cocycle K F 1
    β : CochainComplex.HomComplex.Cochain K G 0
    eq : Eq (HAdd.hAdd (CochainComplex.HomComplex.δ 0 1 β) ((↑α).comp (CochainComp …
    ⊢ Eq (CochainComplex.HomComplex.Cochain.ofHom (CochainComplex.mappingCone.lift …
  -/
  simp only [lift, Cocycle.cochain_ofHom_homOf_eq_coe, liftCocycle_coe]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma lift_f_fst_v (p q : ℤ) (hpq : p + 1 = q) :
    (lift φ α β eq).f p ≫ (fst φ).1.v p q hpq = α.1.v p q hpq := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    α : CochainComplex.HomComplex.Cocycle K F 1
    β : CochainComplex.HomComplex.Cochain K G 0
    eq : Eq (HAdd.hAdd (CochainComplex.HomComplex.δ 0 1 β) ((↑α).comp (CochainComp …
    p q : Int
    hpq : Eq (HAdd.hAdd p 1) q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.lift φ α …
  -/
  simp [lift]
  /-
    🎉 no goals
  -/


lemma lift_fst :
                                                                            /-
                                                                              C : Type u_1
                                                                              inst✝² : CategoryTheory.Category.{u_3, u_1} C
                                                                              inst✝¹ : CategoryTheory.Preadditive C
                                                                              F G : CochainComplex C Int
                                                                              φ : Quiver.Hom F G
                                                                              inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                                                              K : CochainComplex C Int
                                                                              α : CochainComplex.HomComplex.Cocycle K F 1
                                                                              β : CochainComplex.HomComplex.Cochain K G 0
                                                                              eq : Eq (HAdd.hAdd (CochainComplex.HomComplex.δ 0 1 β) ((↑α).comp (CochainComp …
                                                                              ⊢ Eq ((CochainComplex.HomComplex.Cochain.ofHom (CochainComplex.mappingCone.lif …
                                                                            -/
    (Cochain.ofHom (lift φ α β eq)).comp (fst φ).1 (zero_add 1) = α.1 := by simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[reassoc (attr := simp)]
lemma lift_f_snd_v (p q : ℤ) (hpq : p + 0 = q) :
    (lift φ α β eq).f p ≫ (snd φ).v p q hpq = β.v p q hpq := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    α : CochainComplex.HomComplex.Cocycle K F 1
    β : CochainComplex.HomComplex.Cochain K G 0
    eq : Eq (HAdd.hAdd (CochainComplex.HomComplex.δ 0 1 β) ((↑α).comp (CochainComp …
    p q : Int
    hpq : Eq (HAdd.hAdd p 0) q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.lift φ α …
  -/
  obtain rfl : q = p := by omega
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    α : CochainComplex.HomComplex.Cocycle K F 1
    β : CochainComplex.HomComplex.Cochain K G 0
    eq : Eq (HAdd.hAdd (CochainComplex.HomComplex.δ 0 1 β) ((↑α).comp (CochainComp …
    q : Int
    hpq : Eq (HAdd.hAdd q 0) q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.lift φ α …
  -/
  simp [lift]
  /-
    🎉 no goals
  -/


lemma lift_snd :
                                                                        /-
                                                                          C : Type u_1
                                                                          inst✝² : CategoryTheory.Category.{u_3, u_1} C
                                                                          inst✝¹ : CategoryTheory.Preadditive C
                                                                          F G : CochainComplex C Int
                                                                          φ : Quiver.Hom F G
                                                                          inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                                                          K : CochainComplex C Int
                                                                          α : CochainComplex.HomComplex.Cocycle K F 1
                                                                          β : CochainComplex.HomComplex.Cochain K G 0
                                                                          eq : Eq (HAdd.hAdd (CochainComplex.HomComplex.δ 0 1 β) ((↑α).comp (CochainComp …
                                                                          ⊢ Eq ((CochainComplex.HomComplex.Cochain.ofHom (CochainComplex.mappingCone.lif …
                                                                        -/
    (Cochain.ofHom (lift φ α β eq)).comp (snd φ) (zero_add 0) = β := by simp
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


lemma lift_f (p q : ℤ) (hpq : p + 1 = q) :
    (lift φ α β eq).f p = α.1.v p q hpq ≫
                        /-
                          C : Type u_1
                          D : Type u_2
                          inst✝⁴ : CategoryTheory.Category.{?u.222767, u_1} C
                          inst✝³ : CategoryTheory.Category.{?u.222771, u_2} D
                          inst✝² : CategoryTheory.Preadditive C
                          inst✝¹ : CategoryTheory.Preadditive D
                          F G : CochainComplex C Int
                          φ : Quiver.Hom F G
                          inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                          K : CochainComplex C Int
                          α : CochainComplex.HomComplex.Cocycle K F 1
                          β : CochainComplex.HomComplex.Cochain K G 0
                          eq : Eq (HAdd.hAdd (CochainComplex.HomComplex.δ 0 1 β) ((↑α).comp (CochainComp …
                          p q : Int
                          hpq : Eq (HAdd.hAdd p 1) q
                          ⊢ Eq (HAdd.hAdd q (-1)) p
                        -/
      (inl φ).v q p (by omega) + β.v p p (add_zero p) ≫ (inr φ).f p := by
                        /-
                          🎉 no goals
                        -/
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K : CochainComplex C Int
    α : CochainComplex.HomComplex.Cocycle K F 1
    β : CochainComplex.HomComplex.Cochain K G 0
    eq : Eq (HAdd.hAdd (CochainComplex.HomComplex.δ 0 1 β) ((↑α).comp (CochainComp …
    p q : Int
    hpq : Eq (HAdd.hAdd p 1) q
    ⊢ Eq ((CochainComplex.mappingCone.lift φ α β eq).f p) (HAdd.hAdd (CategoryTheo …
  -/
  simp [ext_to_iff _ _ _ hpq]
  /-
    🎉 no goals
  -/


/-- Constructor for homotopies between morphisms to a mapping cone. -/
noncomputable def liftHomotopy {K : CochainComplex C ℤ} (f₁ f₂ : K ⟶ mappingCone φ)
    (α : Cochain K F 0) (β : Cochain K G (-1))
    (h₁ : (Cochain.ofHom f₁).comp (fst φ).1 (zero_add 1) =
      -δ 0 1 α + (Cochain.ofHom f₂).comp (fst φ).1 (zero_add 1))
    (h₂ : (Cochain.ofHom f₁).comp (snd φ) (zero_add 0) =
      δ (-1) 0 β + α.comp (Cochain.ofHom φ) (zero_add 0) +
        (Cochain.ofHom f₂).comp (snd φ) (zero_add 0)) :
    Homotopy f₁ f₂ :=
  (Cochain.equivHomotopy f₁ f₂).symm ⟨liftCochain φ α β (neg_add_cancel 1), by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.229215, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.229219, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Preadditive D
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
      K : CochainComplex C Int
      f₁ f₂ : Quiver.Hom K (CochainComplex.mappingCone φ)
      α : CochainComplex.HomComplex.Cochain K F 0
      β : CochainComplex.HomComplex.Cochain K G (-1)
      h₁ : Eq ((CochainComplex.HomComplex.Cochain.ofHom f₁).comp ↑(CochainComplex.ma …
      h₂ : Eq ((CochainComplex.HomComplex.Cochain.ofHom f₁).comp (CochainComplex.map …
      ⊢ Eq (CochainComplex.HomComplex.Cochain.ofHom f₁) (HAdd.hAdd (CochainComplex.H …
    -/
    simp [δ_liftCochain _ _ _ _ _ (zero_add 1), ext_cochain_to_iff _ _ _ (zero_add 1), h₁, h₂]⟩
    /-
      🎉 no goals
    -/


@[simp]
lemma liftCochain_descCochain :
    (liftCochain φ α β h).comp (descCochain φ α' β' h') hp =
                    /-
                      C : Type u_1
                      D : Type u_2
                      inst✝⁴ : CategoryTheory.Category.{?u.242694, u_1} C
                      inst✝³ : CategoryTheory.Category.{?u.242698, u_2} D
                      inst✝² : CategoryTheory.Preadditive C
                      inst✝¹ : CategoryTheory.Preadditive D
                      F G : CochainComplex C Int
                      φ : Quiver.Hom F G
                      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                      K L : CochainComplex C Int
                      n m : Int
                      α : CochainComplex.HomComplex.Cochain K F m
                      β : CochainComplex.HomComplex.Cochain K G n
                      n' m' : Int
                      α' : CochainComplex.HomComplex.Cochain F L m'
                      β' : CochainComplex.HomComplex.Cochain G L n'
                      h : Eq (HAdd.hAdd n 1) m
                      h' : Eq (HAdd.hAdd m' 1) n'
                      p : Int
                      hp : Eq (HAdd.hAdd n n') p
                      ⊢ Eq (HAdd.hAdd m m') p
                    -/
                    /-
                      🎉 no goals
                    -/
      α.comp α' (by omega) + β.comp β' (by omega) := by
                                           /-
                                             🎉 no goals
                                           -/
  simp [liftCochain, descCochain,
    Cochain.comp_assoc α (inl φ) _ _ (show -1 + n' = m' by omega) (by linarith)]


lemma liftCochain_v_descCochain_v (p₁ p₂ p₃ : ℤ) (h₁₂ : p₁ + n = p₂) (h₂₃ : p₂ + n' = p₃)
    (q : ℤ) (hq : p₁ + m = q) :
    (liftCochain φ α β h).v p₁ p₂ h₁₂ ≫ (descCochain φ α' β' h').v p₂ p₃ h₂₃ =
                                  /-
                                    C : Type u_1
                                    D : Type u_2
                                    inst✝⁴ : CategoryTheory.Category.{?u.248361, u_1} C
                                    inst✝³ : CategoryTheory.Category.{?u.248365, u_2} D
                                    inst✝² : CategoryTheory.Preadditive C
                                    inst✝¹ : CategoryTheory.Preadditive D
                                    F G : CochainComplex C Int
                                    φ : Quiver.Hom F G
                                    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                    K L : CochainComplex C Int
                                    n m : Int
                                    α : CochainComplex.HomComplex.Cochain K F m
                                    β : CochainComplex.HomComplex.Cochain K G n
                                    n' m' : Int
                                    α' : CochainComplex.HomComplex.Cochain F L m'
                                    β' : CochainComplex.HomComplex.Cochain G L n'
                                    h : Eq (HAdd.hAdd n 1) m
                                    h' : Eq (HAdd.hAdd m' 1) n'
                                    p : Int
                                    hp : Eq (HAdd.hAdd n n') p
                                    p₁ p₂ p₃ : Int
                                    h₁₂ : Eq (HAdd.hAdd p₁ n) p₂
                                    h₂₃ : Eq (HAdd.hAdd p₂ n') p₃
                                    q : Int
                                    hq : Eq (HAdd.hAdd p₁ m) q
                                    ⊢ Eq (HAdd.hAdd q m') p₃
                                  -/
      α.v p₁ q hq ≫ α'.v q p₃ (by omega) + β.v p₁ p₂ h₁₂ ≫ β'.v p₂ p₃ h₂₃ := by
                                  /-
                                    🎉 no goals
                                  -/
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝ : HomologicalComplex.HasHomotopyCofiber φ
    K L : CochainComplex C Int
    n m : Int
    α : CochainComplex.HomComplex.Cochain K F m
    β : CochainComplex.HomComplex.Cochain K G n
    n' m' : Int
    α' : CochainComplex.HomComplex.Cochain F L m'
    β' : CochainComplex.HomComplex.Cochain G L n'
    h : Eq (HAdd.hAdd n 1) m
    h' : Eq (HAdd.hAdd m' 1) n'
    p : Int
    hp : Eq (HAdd.hAdd n n') p
    p₁ p₂ p₃ : Int
    h₁₂ : Eq (HAdd.hAdd p₁ n) p₂
    h₂₃ : Eq (HAdd.hAdd p₂ n') p₃
    q : Int
    hq : Eq (HAdd.hAdd p₁ m) q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.liftCoch …
  -/
  have eq := Cochain.congr_v (liftCochain_descCochain φ α β α' β' h h' p hp) p₁ p₃ (by omega)
  simpa only [Cochain.comp_v _ _ hp p₁ p₂ p₃ h₁₂ h₂₃, Cochain.add_v,
    Cochain.comp_v _ _ _ _ _ _ hq (show q + m' = p₃ by omega)] using eq


lemma lift_desc_f {K L : CochainComplex C ℤ} (α : Cocycle K F 1) (β : Cochain K G 0)
    (eq : δ 0 1 β + α.1.comp (Cochain.ofHom φ) (add_zero 1) = 0)
    (α' : Cochain F L (-1)) (β' : G ⟶ L)
    (eq' : δ (-1) 0 α' = Cochain.ofHom (φ ≫ β')) (n n' : ℤ) (hnn' : n + 1 = n') :
    (lift φ α β eq).f n ≫ (desc φ α' β' eq').f n =
                                    /-
                                      C : Type u_1
                                      D : Type u_2
                                      inst✝⁴ : CategoryTheory.Category.{?u.256790, u_1} C
                                      inst✝³ : CategoryTheory.Category.{?u.256794, u_2} D
                                      inst✝² : CategoryTheory.Preadditive C
                                      inst✝¹ : CategoryTheory.Preadditive D
                                      F G : CochainComplex C Int
                                      φ : Quiver.Hom F G
                                      inst✝ : HomologicalComplex.HasHomotopyCofiber φ
                                      K L : CochainComplex C Int
                                      α : CochainComplex.HomComplex.Cocycle K F 1
                                      β : CochainComplex.HomComplex.Cochain K G 0
                                      eq : Eq (HAdd.hAdd (CochainComplex.HomComplex.δ 0 1 β) ((↑α).comp (CochainComp …
                                      α' : CochainComplex.HomComplex.Cochain F L (-1)
                                      β' : Quiver.Hom G L
                                      eq' : Eq (CochainComplex.HomComplex.δ (-1) 0 α') (CochainComplex.HomComplex.Co …
                                      n n' : Int
                                      hnn' : Eq (HAdd.hAdd n 1) n'
                                      ⊢ Eq (HAdd.hAdd n' (-1)) n
                                    -/
    α.1.v n n' hnn' ≫ α'.v n' n (by omega) + β.v n n (add_zero n) ≫ β'.f n := by
                                    /-
                                      🎉 no goals
                                    -/
  simp only [lift, desc, Cocycle.homOf_f, liftCocycle_coe, descCocycle_coe, Cocycle.ofHom_coe,
    liftCochain_v_descCochain_v φ α.1 β α' (Cochain.ofHom β') (zero_add 1) (neg_add_cancel 1) 0
    (add_zero 0) n n n (add_zero n) (add_zero n) n' hnn', Cochain.ofHom_v]



/-- If `H : C ⥤ D` is an additive functor and `φ` is a morphism of cochain complexes
in `C`, this is the comparison isomorphism (in each degree `n`) between the image
by `H` of `mappingCone φ` and the mapping cone of the image by `H` of `φ`.
It is an auxiliary definition for `mapHomologicalComplexXIso` and
`mapHomologicalComplexIso`. This definition takes an extra
parameter `m : ℤ` such that `n + 1 = m` which may help getting better
definitional properties. See also the equational lemma `mapHomologicalComplexXIso_eq`. -/
@[simps]
noncomputable def mapHomologicalComplexXIso' (n m : ℤ) (hnm : n + 1 = m) :
    ((H.mapHomologicalComplex (ComplexShape.up ℤ)).obj (mappingCone φ)).X n ≅
      (mappingCone ((H.mapHomologicalComplex (ComplexShape.up ℤ)).map φ)).X n where
                                    /-
                                      C : Type u_1
                                      D : Type u_2
                                      inst✝⁶ : CategoryTheory.Category.{?u.263879, u_1} C
                                      inst✝⁵ : CategoryTheory.Category.{?u.263883, u_2} D
                                      inst✝⁴ : CategoryTheory.Preadditive C
                                      inst✝³ : CategoryTheory.Preadditive D
                                      F G : CochainComplex C Int
                                      φ : Quiver.Hom F G
                                      inst✝² : HomologicalComplex.HasHomotopyCofiber φ
                                      H : CategoryTheory.Functor C D
                                      inst✝¹ : H.Additive
                                      inst✝ : HomologicalComplex.HasHomotopyCofiber ((H.mapHomologicalComplex (Compl …
                                      n m : Int
                                      hnm : Eq (HAdd.hAdd n 1) m
                                      ⊢ Eq (HAdd.hAdd n 1) m
                                    -/
  hom := H.map ((fst φ).1.v n m (by omega)) ≫
                                    /-
                                      🎉 no goals
                                    -/
                                                                            /-
                                                                              C : Type u_1
                                                                              D : Type u_2
                                                                              inst✝⁶ : CategoryTheory.Category.{?u.263879, u_1} C
                                                                              inst✝⁵ : CategoryTheory.Category.{?u.263883, u_2} D
                                                                              inst✝⁴ : CategoryTheory.Preadditive C
                                                                              inst✝³ : CategoryTheory.Preadditive D
                                                                              F G : CochainComplex C Int
                                                                              φ : Quiver.Hom F G
                                                                              inst✝² : HomologicalComplex.HasHomotopyCofiber φ
                                                                              H : CategoryTheory.Functor C D
                                                                              inst✝¹ : H.Additive
                                                                              inst✝ : HomologicalComplex.HasHomotopyCofiber ((H.mapHomologicalComplex (Compl …
                                                                              n m : Int
                                                                              hnm : Eq (HAdd.hAdd n 1) m
                                                                              ⊢ Eq (HAdd.hAdd m (-1)) n
                                                                            -/
      (inl ((H.mapHomologicalComplex (ComplexShape.up ℤ)).map φ)).v m n (by omega) +
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
      H.map ((snd φ).v n n (add_zero n)) ≫
        (inr ((H.mapHomologicalComplex (ComplexShape.up ℤ)).map φ)).f n
                                                                                 /-
                                                                                   C : Type u_1
                                                                                   D : Type u_2
                                                                                   inst✝⁶ : CategoryTheory.Category.{?u.263879, u_1} C
                                                                                   inst✝⁵ : CategoryTheory.Category.{?u.263883, u_2} D
                                                                                   inst✝⁴ : CategoryTheory.Preadditive C
                                                                                   inst✝³ : CategoryTheory.Preadditive D
                                                                                   F G : CochainComplex C Int
                                                                                   φ : Quiver.Hom F G
                                                                                   inst✝² : HomologicalComplex.HasHomotopyCofiber φ
                                                                                   H : CategoryTheory.Functor C D
                                                                                   inst✝¹ : H.Additive
                                                                                   inst✝ : HomologicalComplex.HasHomotopyCofiber ((H.mapHomologicalComplex (Compl …
                                                                                   n m : Int
                                                                                   hnm : Eq (HAdd.hAdd n 1) m
                                                                                   ⊢ Eq (HAdd.hAdd n 1) m
                                                                                 -/
  inv := (fst ((H.mapHomologicalComplex (ComplexShape.up ℤ)).map φ)).1.v n m (by omega) ≫
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
                               /-
                                 C : Type u_1
                                 D : Type u_2
                                 inst✝⁶ : CategoryTheory.Category.{?u.263879, u_1} C
                                 inst✝⁵ : CategoryTheory.Category.{?u.263883, u_2} D
                                 inst✝⁴ : CategoryTheory.Preadditive C
                                 inst✝³ : CategoryTheory.Preadditive D
                                 F G : CochainComplex C Int
                                 φ : Quiver.Hom F G
                                 inst✝² : HomologicalComplex.HasHomotopyCofiber φ
                                 H : CategoryTheory.Functor C D
                                 inst✝¹ : H.Additive
                                 inst✝ : HomologicalComplex.HasHomotopyCofiber ((H.mapHomologicalComplex (Compl …
                                 n m : Int
                                 hnm : Eq (HAdd.hAdd n 1) m
                                 ⊢ Eq (HAdd.hAdd m (-1)) n
                               -/
      H.map ((inl φ).v m n (by omega)) +
                               /-
                                 🎉 no goals
                               -/
      (snd ((H.mapHomologicalComplex (ComplexShape.up ℤ)).map φ)).v n n (add_zero n) ≫
        H.map ((inr φ).f n)
  hom_inv_id := by
    simp only [Functor.mapHomologicalComplex_obj_X, comp_add, add_comp, assoc,
      inl_v_fst_v_assoc, inr_f_fst_v_assoc, zero_comp, comp_zero, add_zero,
      inl_v_snd_v_assoc, inr_f_snd_v_assoc, zero_add, ← Functor.map_comp, ← Functor.map_add]
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.263879, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.263883, u_2} D
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Preadditive D
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝² : HomologicalComplex.HasHomotopyCofiber φ
      H : CategoryTheory.Functor C D
      inst✝¹ : H.Additive
      inst✝ : HomologicalComplex.HasHomotopyCofiber ((H.mapHomologicalComplex (Compl …
      n m : Int
      hnm : Eq (HAdd.hAdd n 1) m
      ⊢ Eq (H.map (HAdd.hAdd (CategoryTheory.CategoryStruct.comp ((↑(CochainComplex. …
    -/
    rw [← H.map_id]
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.263879, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.263883, u_2} D
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Preadditive D
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝² : HomologicalComplex.HasHomotopyCofiber φ
      H : CategoryTheory.Functor C D
      inst✝¹ : H.Additive
      inst✝ : HomologicalComplex.HasHomotopyCofiber ((H.mapHomologicalComplex (Compl …
      n m : Int
      hnm : Eq (HAdd.hAdd n 1) m
      ⊢ Eq (H.map (HAdd.hAdd (CategoryTheory.CategoryStruct.comp ((↑(CochainComplex. …
    -/
    congr 1
    /-
      case e_a
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.263879, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.263883, u_2} D
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Preadditive D
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝² : HomologicalComplex.HasHomotopyCofiber φ
      H : CategoryTheory.Functor C D
      inst✝¹ : H.Additive
      inst✝ : HomologicalComplex.HasHomotopyCofiber ((H.mapHomologicalComplex (Compl …
      n m : Int
      hnm : Eq (HAdd.hAdd n 1) m
      ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp ((↑(CochainComplex.mapping …
    -/
    simp [ext_from_iff  _ _ _ hnm]
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    simp only [Functor.mapHomologicalComplex_obj_X, comp_add, add_comp, assoc,
      ← H.map_comp_assoc, inl_v_fst_v, CategoryTheory.Functor.map_id, id_comp, inr_f_fst_v,
      inl_v_snd_v, inr_f_snd_v]
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.263879, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.263883, u_2} D
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Preadditive D
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝² : HomologicalComplex.HasHomotopyCofiber φ
      H : CategoryTheory.Functor C D
      inst✝¹ : H.Additive
      inst✝ : HomologicalComplex.HasHomotopyCofiber ((H.mapHomologicalComplex (Compl …
      n m : Int
      hnm : Eq (HAdd.hAdd n 1) m
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp ((↑(CochainComp …
    -/
    simp [ext_from_iff _ _ _ hnm]
    /-
      🎉 no goals
    -/


/-- If `H : C ⥤ D` is an additive functor and `φ` is a morphism of cochain complexes
in `C`, this is the comparison isomorphism (in each degree) between the image
by `H` of `mappingCone φ` and the mapping cone of the image by `H` of `φ`. -/
noncomputable def mapHomologicalComplexXIso (n : ℤ) :
    ((H.mapHomologicalComplex (ComplexShape.up ℤ)).obj (mappingCone φ)).X n ≅
      (mappingCone ((H.mapHomologicalComplex (ComplexShape.up ℤ)).map φ)).X n :=
  mapHomologicalComplexXIso' φ H n (n + 1) rfl


lemma mapHomologicalComplexXIso_eq (n m : ℤ) (hnm : n + 1 = m) :
    mapHomologicalComplexXIso φ H n = mapHomologicalComplexXIso' φ H n m hnm := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Preadditive D
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝² : HomologicalComplex.HasHomotopyCofiber φ
    H : CategoryTheory.Functor C D
    inst✝¹ : H.Additive
    inst✝ : HomologicalComplex.HasHomotopyCofiber ((H.mapHomologicalComplex (Compl …
    n m : Int
    hnm : Eq (HAdd.hAdd n 1) m
    ⊢ Eq (CochainComplex.mappingCone.mapHomologicalComplexXIso φ H n) (CochainComp …
  -/
  subst hnm
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Preadditive D
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝² : HomologicalComplex.HasHomotopyCofiber φ
    H : CategoryTheory.Functor C D
    inst✝¹ : H.Additive
    inst✝ : HomologicalComplex.HasHomotopyCofiber ((H.mapHomologicalComplex (Compl …
    n : Int
    ⊢ Eq (CochainComplex.mappingCone.mapHomologicalComplexXIso φ H n) (CochainComp …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `H : C ⥤ D` is an additive functor and `φ` is a morphism of cochain complexes
in `C`, this is the comparison isomorphism between the image by `H`
of `mappingCone φ` and the mapping cone of the image by `H` of `φ`. -/
noncomputable def mapHomologicalComplexIso :
    (H.mapHomologicalComplex _).obj (mappingCone φ) ≅
      mappingCone ((H.mapHomologicalComplex _).map φ) :=
  HomologicalComplex.Hom.isoOfComponents (mapHomologicalComplexXIso φ H) (by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.285004, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.285008, u_2} D
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Preadditive D
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝² : HomologicalComplex.HasHomotopyCofiber φ
      H : CategoryTheory.Functor C D
      inst✝¹ : H.Additive
      inst✝ : HomologicalComplex.HasHomotopyCofiber ((H.mapHomologicalComplex (Compl …
      ⊢ ∀ (i j : Int), (ComplexShape.up Int).Rel i j → Eq (CategoryTheory.CategorySt …
    -/
    rintro n _ rfl
    rw [ext_to_iff _ _ (n + 2) (by omega), assoc, assoc, d_fst_v _ _ _ _ rfl,
      assoc, assoc, d_snd_v _ _ _ rfl]
    simp only [mapHomologicalComplexXIso_eq φ H n (n + 1) rfl,
      mapHomologicalComplexXIso_eq φ H (n + 1) (n + 2) (by omega),
      mapHomologicalComplexXIso'_hom, mapHomologicalComplexXIso'_hom]
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.285004, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.285008, u_2} D
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Preadditive D
      F G : CochainComplex C Int
      φ : Quiver.Hom F G
      inst✝² : HomologicalComplex.HasHomotopyCofiber φ
      H : CategoryTheory.Functor C D
      inst✝¹ : H.Additive
      inst✝ : HomologicalComplex.HasHomotopyCofiber ((H.mapHomologicalComplex (Compl …
      n : Int
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd (CategoryTheory.Categ …
    -/
    constructor
      /-
        case left
        C : Type u_1
        D : Type u_2
        inst✝⁶ : CategoryTheory.Category.{?u.285004, u_1} C
        inst✝⁵ : CategoryTheory.Category.{?u.285008, u_2} D
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Preadditive D
        F G : CochainComplex C Int
        φ : Quiver.Hom F G
        inst✝² : HomologicalComplex.HasHomotopyCofiber φ
        H : CategoryTheory.Functor C D
        inst✝¹ : H.Additive
        inst✝ : HomologicalComplex.HasHomotopyCofiber ((H.mapHomologicalComplex (Compl …
        n : Int
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd (CategoryTheory.CategorySt …
      -/
    · dsimp
      simp only [Functor.mapHomologicalComplex_obj_X, Functor.mapHomologicalComplex_obj_d,
        comp_neg, add_comp, assoc, inl_v_fst_v_assoc, inr_f_fst_v_assoc, zero_comp,
        comp_zero, add_zero, comp_add, inl_v_fst_v, comp_id, inr_f_fst_v, ← H.map_comp,
        d_fst_v φ n (n + 1) (n + 2) rfl (by omega), Functor.map_neg]
      /-
        case right
        C : Type u_1
        D : Type u_2
        inst✝⁶ : CategoryTheory.Category.{?u.285004, u_1} C
        inst✝⁵ : CategoryTheory.Category.{?u.285008, u_2} D
        inst✝⁴ : CategoryTheory.Preadditive C
        inst✝³ : CategoryTheory.Preadditive D
        F G : CochainComplex C Int
        φ : Quiver.Hom F G
        inst✝² : HomologicalComplex.HasHomotopyCofiber φ
        H : CategoryTheory.Functor C D
        inst✝¹ : H.Additive
        inst✝ : HomologicalComplex.HasHomotopyCofiber ((H.mapHomologicalComplex (Compl …
        n : Int
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd (CategoryTheory.CategorySt …
      -/
    · dsimp
      simp only [comp_add, add_comp, assoc, inl_v_fst_v_assoc, inr_f_fst_v_assoc,
        Functor.mapHomologicalComplex_obj_X, zero_comp, comp_zero, add_zero, inl_v_snd_v_assoc,
        inr_f_snd_v_assoc, zero_add, inl_v_snd_v, inr_f_snd_v, comp_id, ← H.map_comp,
        d_snd_v φ n (n + 1) rfl, Functor.map_add])


lemma map_inr :
    (H.mapHomologicalComplex (ComplexShape.up ℤ)).map (inr φ) ≫
      (mapHomologicalComplexIso φ H).hom =
    inr ((Functor.mapHomologicalComplex H (ComplexShape.up ℤ)).map φ) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Preadditive D
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝² : HomologicalComplex.HasHomotopyCofiber φ
    H : CategoryTheory.Functor C D
    inst✝¹ : H.Additive
    inst✝ : HomologicalComplex.HasHomotopyCofiber ((H.mapHomologicalComplex (Compl …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((H.mapHomologicalComplex (ComplexSha …
  -/
  ext n
  /-
    case h
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Preadditive D
    F G : CochainComplex C Int
    φ : Quiver.Hom F G
    inst✝² : HomologicalComplex.HasHomotopyCofiber φ
    H : CategoryTheory.Functor C D
    inst✝¹ : H.Additive
    inst✝ : HomologicalComplex.HasHomotopyCofiber ((H.mapHomologicalComplex (Compl …
    n : Int
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((H.mapHomologicalComplex (ComplexSh …
  -/
  dsimp [mapHomologicalComplexIso]
  simp only [mapHomologicalComplexXIso_eq φ H n (n + 1) rfl, mappingCone.ext_to_iff _ _ _ rfl,
    Functor.mapHomologicalComplex_obj_X, mapHomologicalComplexXIso'_hom, comp_add,
    add_comp, assoc, inl_v_fst_v, comp_id, inr_f_fst_v, comp_zero, add_zero, inl_v_snd_v,
    inr_f_snd_v, zero_add, ← H.map_comp, H.map_zero, H.map_id, and_self]



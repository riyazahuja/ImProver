/-- The `1`-cocycle attached to a degreewise split short exact sequence of cochain complexes. -/
def cocycleOfDegreewiseSplit : Cocycle S.X₃ S.X₁ 1 :=
  Cocycle.mk
                                                                     /-
                                                                       C : Type u_1
                                                                       inst✝¹ : CategoryTheory.Category.{?u.397, u_1} C
                                                                       inst✝ : CategoryTheory.Preadditive C
                                                                       S : CategoryTheory.ShortComplex (CochainComplex C Int)
                                                                       σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
                                                                       ⊢ Eq (HAdd.hAdd 1 1) 2
                                                                     -/
    (Cochain.mk (fun p q _ => (σ p).s ≫ S.X₂.d p q ≫ (σ q).r)) 2 (by omega) (by
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
      /-
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.397, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        S : CategoryTheory.ShortComplex (CochainComplex C Int)
        σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
        ⊢ Eq (CochainComplex.HomComplex.δ 1 2 (CochainComplex.HomComplex.Cochain.mk fu …
      -/
      ext p _ rfl
      /-
        case h
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.397, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        S : CategoryTheory.ShortComplex (CochainComplex C Int)
        σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
        p : Int
        ⊢ Eq ((CochainComplex.HomComplex.δ 1 2 (CochainComplex.HomComplex.Cochain.mk f …
      -/
      have := mono_of_mono_fac (σ (p + 2)).f_r
      /-
        case h
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.397, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        S : CategoryTheory.ShortComplex (CochainComplex C Int)
        σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
        p : Int
        this : CategoryTheory.Mono (S.map (HomologicalComplex.eval C (ComplexShape.up  …
        ⊢ Eq ((CochainComplex.HomComplex.δ 1 2 (CochainComplex.HomComplex.Cochain.mk f …
      -/
      have r_f := fun n => (σ n).r_f
      /-
        case h
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.397, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        S : CategoryTheory.ShortComplex (CochainComplex C Int)
        σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
        p : Int
        this : CategoryTheory.Mono (S.map (HomologicalComplex.eval C (ComplexShape.up  …
        r_f : ∀ (n : Int), Eq (CategoryTheory.CategoryStruct.comp (σ n).r (S.map (Homo …
        ⊢ Eq ((CochainComplex.HomComplex.δ 1 2 (CochainComplex.HomComplex.Cochain.mk f …
      -/
      have s_g := fun n => (σ n).s_g
      /-
        case h
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.397, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        S : CategoryTheory.ShortComplex (CochainComplex C Int)
        σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
        p : Int
        this : CategoryTheory.Mono (S.map (HomologicalComplex.eval C (ComplexShape.up  …
        r_f : ∀ (n : Int), Eq (CategoryTheory.CategoryStruct.comp (σ n).r (S.map (Homo …
        s_g : ∀ (n : Int), Eq (CategoryTheory.CategoryStruct.comp (σ n).s (S.map (Homo …
        ⊢ Eq ((CochainComplex.HomComplex.δ 1 2 (CochainComplex.HomComplex.Cochain.mk f …
      -/
      dsimp at this r_f s_g ⊢
      rw [δ_v 1 2 (by omega) _ p (p + 2) (by omega) (p + 1) (p + 1)
        (by omega) (by omega), Cochain.mk_v, Cochain.mk_v,
        show Int.negOnePow 2 = 1 by rfl, one_smul, assoc, assoc,
        ← cancel_mono (S.f.f (p + 2)), add_comp, assoc, assoc, assoc,
        assoc, assoc, assoc, zero_comp, ← S.f.comm, reassoc_of% (r_f (p + 1)),
        sub_comp, comp_sub, comp_sub, assoc, id_comp, d_comp_d, comp_zero, zero_sub,
        ← S.g.comm_assoc, reassoc_of% (s_g p), r_f (p + 2), comp_sub, comp_sub, comp_id,
        comp_sub, ← S.g.comm_assoc, reassoc_of% (s_g (p + 1)), d_comp_d_assoc, zero_comp,
        sub_zero, neg_add_cancel])


/-- The canonical morphism `S.X₃ ⟶ S.X₁⟦(1 : ℤ)⟧` attached to a degreewise split
short exact sequence of cochain complexes. -/
def homOfDegreewiseSplit : S.X₃ ⟶ S.X₁⟦(1 : ℤ)⟧ :=
  ((Cocycle.equivHom _ _).symm ((cocycleOfDegreewiseSplit S σ).rightShift 1 0 (zero_add 1)))


@[simp]
lemma homOfDegreewiseSplit_f (n : ℤ) :
    (homOfDegreewiseSplit S σ).f n =
      (cocycleOfDegreewiseSplit S σ).1.v n (n + 1) rfl := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
    n : Int
    ⊢ Eq ((CochainComplex.homOfDegreewiseSplit S σ).f n) ((↑(CochainComplex.cocycl …
  -/
  simp [homOfDegreewiseSplit, Cochain.rightShift_v _ _ _ _ _ _ _ _ rfl]
  /-
    🎉 no goals
  -/


/-- The triangle in `CochainComplex C ℤ` attached to a degreewise split short exact sequence
of cochain complexes. -/
@[simps! obj₁ obj₂ obj₃ mor₁ mor₂ mor₃]
def triangleOfDegreewiseSplit : Triangle (CochainComplex C ℤ) :=
  Triangle.mk S.f S.g (homOfDegreewiseSplit S σ)


/-- The (distinguished) triangle in `HomotopyCategory C (ComplexShape.up ℤ)` attached to a
degreewise split short exact sequence of cochain complexes. -/
noncomputable abbrev trianglehOfDegreewiseSplit :
    Triangle (HomotopyCategory C (ComplexShape.up ℤ)) :=
  (HomotopyCategory.quotient C (ComplexShape.up ℤ)).mapTriangle.obj (triangleOfDegreewiseSplit S σ)


/-- The canonical isomorphism `(mappingCone (homOfDegreewiseSplit S σ)).X p ≅ S.X₂.X q`
when `p + 1 = q`. -/
noncomputable def mappingConeHomOfDegreewiseSplitXIso (p q : ℤ) (hpq : p + 1 = q) :
    (mappingCone (homOfDegreewiseSplit S σ)).X p ≅ S.X₂.X q where
  hom := (mappingCone.fst (homOfDegreewiseSplit S σ)).1.v p q hpq ≫ (σ q).s -
    (mappingCone.snd (homOfDegreewiseSplit S σ)).v p p (add_zero p) ≫
         /-
           C : Type u_1
           inst✝² : CategoryTheory.Category.{?u.59984, u_1} C
           inst✝¹ : CategoryTheory.Preadditive C
           S : CategoryTheory.ShortComplex (CochainComplex C Int)
           σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
           inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
           p q : Int
           hpq : Eq (HAdd.hAdd p 1) q
           ⊢ Quiver.Hom (((CategoryTheory.shiftFunctor (CochainComplex C Int) 1).obj S.X₁ …
         -/
      by exact (Cochain.ofHom S.f).v (p + 1) q (by omega)
         /-
           🎉 no goals
         -/
                                                                          /-
                                                                            C : Type u_1
                                                                            inst✝² : CategoryTheory.Category.{?u.59984, u_1} C
                                                                            inst✝¹ : CategoryTheory.Preadditive C
                                                                            S : CategoryTheory.ShortComplex (CochainComplex C Int)
                                                                            σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
                                                                            inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
                                                                            p q : Int
                                                                            hpq : Eq (HAdd.hAdd p 1) q
                                                                            ⊢ Eq (HAdd.hAdd q (-1)) p
                                                                          -/
  inv := S.g.f q ≫ (mappingCone.inl (homOfDegreewiseSplit S σ)).v q p (by omega) -
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    by exact (σ q).r ≫ (S.X₁.XIsoOfEq hpq.symm).hom ≫
      (mappingCone.inr (homOfDegreewiseSplit S σ)).f p
  hom_inv_id := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.59984, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      p q : Int
      hpq : Eq (HAdd.hAdd p 1) q
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (CategoryTheory.CategorySt …
    -/
    subst hpq
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.59984, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      p : Int
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (CategoryTheory.CategorySt …
    -/
    have s_g := (σ (p + 1)).s_g
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.59984, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      p : Int
      s_g : Eq (CategoryTheory.CategoryStruct.comp (σ (HAdd.hAdd p 1)).s (S.map (Hom …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (CategoryTheory.CategorySt …
    -/
    have f_r := (σ (p + 1)).f_r
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.59984, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      p : Int
      s_g : Eq (CategoryTheory.CategoryStruct.comp (σ (HAdd.hAdd p 1)).s (S.map (Hom …
      f_r : Eq (CategoryTheory.CategoryStruct.comp (S.map (HomologicalComplex.eval C …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (CategoryTheory.CategorySt …
    -/
    dsimp at s_g f_r ⊢
    -- the following list of lemmas was obtained by doing
    -- simp? [mappingCone.ext_from_iff _ (p + 1) _ rfl, reassoc_of% f_r, reassoc_of% s_g]
    -- which may require increasing maximum heart beats
    simp only [Cochain.ofHom_v, Int.reduceNeg, id_comp, comp_sub, sub_comp, assoc,
        reassoc_of% s_g, ShortComplex.Splitting.s_r_assoc, ShortComplex.map_X₃, eval_obj,
        ShortComplex.map_X₁, zero_comp, comp_zero, reassoc_of% f_r, zero_sub, sub_neg_eq_add,
        mappingCone.ext_from_iff _ (p + 1) _ rfl, comp_add, mappingCone.inl_v_fst_v_assoc,
        mappingCone.inl_v_snd_v_assoc, shiftFunctor_obj_X', sub_zero, add_zero, comp_id,
        mappingCone.inr_f_fst_v_assoc, mappingCone.inr_f_snd_v_assoc, add_left_eq_self, neg_eq_zero,
        true_and]
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.59984, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      p : Int
      s_g : Eq (CategoryTheory.CategoryStruct.comp (σ (HAdd.hAdd p 1)).s (S.g.f (HAd …
      f_r : Eq (CategoryTheory.CategoryStruct.comp (S.f.f (HAdd.hAdd p 1)) (σ (HAdd. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.f.f (HAdd.hAdd p 1)) (CategoryTheo …
    -/
    rw [← comp_f_assoc, S.zero, zero_f, zero_comp]
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.59984, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      p q : Int
      hpq : Eq (HAdd.hAdd p 1) q
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (CategoryTheory.CategorySt …
    -/
    subst hpq
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.59984, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      p : Int
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (CategoryTheory.CategorySt …
    -/
    have h := (σ (p + 1)).id
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.59984, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      p : Int
      h : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (σ (HAdd.hAdd p 1)).r (S …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (CategoryTheory.CategorySt …
    -/
    dsimp at h ⊢
    simp only [id_comp, Cochain.ofHom_v, comp_sub, sub_comp, assoc, mappingCone.inl_v_fst_v_assoc,
      mappingCone.inr_f_fst_v_assoc, shiftFunctor_obj_X', zero_comp, comp_zero, sub_zero,
      mappingCone.inl_v_snd_v_assoc, mappingCone.inr_f_snd_v_assoc, zero_sub, sub_neg_eq_add, ← h]
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.59984, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      p : Int
      h : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (σ (HAdd.hAdd p 1)).r (S …
      ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (S.g.f (HAdd.hAdd p 1)) (σ …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/


/-- The canonical isomorphism `mappingCone (homOfDegreewiseSplit S σ) ≅ S.X₂⟦(1 : ℤ)⟧`. -/
@[simps!]
noncomputable def mappingConeHomOfDegreewiseSplitIso :
    mappingCone (homOfDegreewiseSplit S σ) ≅ S.X₂⟦(1 : ℤ)⟧ :=
  Hom.isoOfComponents (fun p => mappingConeHomOfDegreewiseSplitXIso S σ p _ rfl) (by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.86087, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      ⊢ ∀ (i j : Int), (ComplexShape.up Int).Rel i j → Eq (CategoryTheory.CategorySt …
    -/
    rintro p _ rfl
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.86087, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      p : Int
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun p => CochainComplex.mappingCone …
    -/
    have r_f := (σ (p + 1 + 1)).r_f
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.86087, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      p : Int
      r_f : Eq (CategoryTheory.CategoryStruct.comp (σ (HAdd.hAdd (HAdd.hAdd p 1) 1)) …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun p => CochainComplex.mappingCone …
    -/
    have s_g := (σ (p + 1)).s_g
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.86087, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      p : Int
      r_f : Eq (CategoryTheory.CategoryStruct.comp (σ (HAdd.hAdd (HAdd.hAdd p 1) 1)) …
      s_g : Eq (CategoryTheory.CategoryStruct.comp (σ (HAdd.hAdd p 1)).s (S.map (Hom …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun p => CochainComplex.mappingCone …
    -/
    dsimp at r_f s_g ⊢
    simp only [mappingConeHomOfDegreewiseSplitXIso, mappingCone.ext_from_iff _ _ _ rfl,
      mappingCone.inl_v_d_assoc _ (p + 1) _ (p + 1 + 1) (by linarith) (by omega),
      cocycleOfDegreewiseSplit, r_f, Int.reduceNeg, Cochain.ofHom_v, sub_comp, assoc,
      Hom.comm, comp_sub, mappingCone.inl_v_fst_v_assoc, mappingCone.inl_v_snd_v_assoc,
      shiftFunctor_obj_X', zero_comp, sub_zero, homOfDegreewiseSplit_f,
      mappingCone.inr_f_fst_v_assoc, comp_zero, zero_sub, mappingCone.inr_f_snd_v_assoc,
      neg_neg, mappingCone.inr_f_d_assoc, shiftFunctor_obj_d',
      Int.negOnePow_one, neg_comp, sub_neg_eq_add, zero_add, and_true,
      Units.neg_smul, one_smul, comp_neg, ShortComplex.map_X₂, eval_obj, Cocycle.mk_coe,
      Cochain.mk_v]
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.86087, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      p : Int
      r_f : Eq (CategoryTheory.CategoryStruct.comp (σ (HAdd.hAdd (HAdd.hAdd p 1) 1)) …
      s_g : Eq (CategoryTheory.CategoryStruct.comp (σ (HAdd.hAdd p 1)).s (S.g.f (HAd …
      ⊢ Eq (Neg.neg (CategoryTheory.CategoryStruct.comp (σ (HAdd.hAdd p 1)).s (S.X₂. …
    -/
    simp only [← S.g.comm_assoc, reassoc_of% s_g, comp_id]
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.86087, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      p : Int
      r_f : Eq (CategoryTheory.CategoryStruct.comp (σ (HAdd.hAdd (HAdd.hAdd p 1) 1)) …
      s_g : Eq (CategoryTheory.CategoryStruct.comp (σ (HAdd.hAdd p 1)).s (S.g.f (HAd …
      ⊢ Eq (Neg.neg (CategoryTheory.CategoryStruct.comp (σ (HAdd.hAdd p 1)).s (S.X₂. …
    -/
    /-
      🎉 no goals
    -/
    abel)
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
lemma shift_f_comp_mappingConeHomOfDegreewiseSplitIso_inv :
    S.f⟦(1 : ℤ)⟧' ≫ (mappingConeHomOfDegreewiseSplitIso S σ).inv = -mappingCone.inr _ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor (Cochai …
  -/
  ext n
  /-
    case h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    n : Int
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor (Cocha …
  -/
  have h := (σ (n + 1)).f_r
  /-
    case h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    n : Int
    h : Eq (CategoryTheory.CategoryStruct.comp (S.map (HomologicalComplex.eval C ( …
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor (Cocha …
  -/
  dsimp at h
  /-
    case h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    n : Int
    h : Eq (CategoryTheory.CategoryStruct.comp (S.f.f (HAdd.hAdd n 1)) (σ (HAdd.hA …
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor (Cocha …
  -/
  dsimp [mappingConeHomOfDegreewiseSplitXIso]
  /-
    case h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    n : Int
    h : Eq (CategoryTheory.CategoryStruct.comp (S.f.f (HAdd.hAdd n 1)) (σ (HAdd.hA …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.f.f (HAdd.hAdd n 1)) (HSub.hSub (C …
  -/
  rw [id_comp, comp_sub, ← comp_f_assoc, S.zero, zero_f, zero_comp, zero_sub, reassoc_of% h]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma mappingConeHomOfDegreewiseSplitIso_inv_comp_triangle_mor₃ :
    (mappingConeHomOfDegreewiseSplitIso S σ).inv ≫
      (mappingCone.triangle (homOfDegreewiseSplit S σ)).mor₃ = -S.g⟦(1 : ℤ)⟧' := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CochainComplex.mappingConeHomOfDegre …
  -/
  ext n
  /-
    case h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    n : Int
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CochainComplex.mappingConeHomOfDegr …
  -/
  dsimp [mappingConeHomOfDegreewiseSplitXIso]
  simp only [Int.reduceNeg, id_comp, sub_comp, assoc, mappingCone.inl_v_triangle_mor₃_f,
    shiftFunctor_obj_X, shiftFunctorObjXIso, XIsoOfEq_rfl, Iso.refl_inv, comp_neg, comp_id,
    mappingCone.inr_f_triangle_mor₃_f, comp_zero, sub_zero]


/-- The canonical isomorphism of triangles
`(triangleOfDegreewiseSplit S σ).rotate.rotate ≅ mappingCone.triangle (homOfDegreewiseSplit S σ)`
when `S` is a degreewise split short exact sequence of cochain complexes. -/
noncomputable def triangleOfDegreewiseSplitRotateRotateIso :
    (triangleOfDegreewiseSplit S σ).rotate.rotate ≅
      mappingCone.triangle (homOfDegreewiseSplit S σ) :=
  Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _) (mappingConeHomOfDegreewiseSplitIso S σ).symm
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.176788, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          S : CategoryTheory.ShortComplex (CochainComplex C Int)
          σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
          inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CochainComplex.triangleOfDegreewiseS …
        -/
    (by dsimp; simp only [comp_id, id_comp])
               /-
                 🎉 no goals
               -/
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.176788, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          S : CategoryTheory.ShortComplex (CochainComplex C Int)
          σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
          inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CochainComplex.triangleOfDegreewiseS …
        -/
    (by dsimp; simp only [neg_comp, shift_f_comp_mappingConeHomOfDegreewiseSplitIso_inv,
      shiftFunctor_obj_X', neg_neg, id_comp])
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.176788, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          S : CategoryTheory.ShortComplex (CochainComplex C Int)
          σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
          inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CochainComplex.triangleOfDegreewiseS …
        -/
    (by dsimp; simp only [CategoryTheory.Functor.map_id, comp_id,
      mappingConeHomOfDegreewiseSplitIso_inv_comp_triangle_mor₃])


/-- The canonical isomorphism between `(trianglehOfDegreewiseSplit S σ).rotate.rotate` and
`mappingCone.triangleh (homOfDegreewiseSplit S σ)` when `S` is a degreewise split
short exact sequence of cochain complexes. -/
noncomputable def trianglehOfDegreewiseSplitRotateRotateIso :
    (trianglehOfDegreewiseSplit S σ).rotate.rotate ≅
      mappingCone.triangleh (homOfDegreewiseSplit S σ) :=
  (rotate _).mapIso ((HomotopyCategory.quotient _ _).mapTriangleRotateIso.app _) ≪≫
    (HomotopyCategory.quotient _ _).mapTriangleRotateIso.app _ ≪≫
    (HomotopyCategory.quotient _ _).mapTriangle.mapIso
      (triangleOfDegreewiseSplitRotateRotateIso S σ)


/-- Given a morphism of cochain complexes `φ`, this is the short complex
given by `(triangle φ).rotate`. -/
@[simps]
noncomputable def triangleRotateShortComplex : ShortComplex (CochainComplex C ℤ) :=
                                                                        /-
                                                                          C : Type u_1
                                                                          inst✝² : CategoryTheory.Category.{?u.202239, u_1} C
                                                                          inst✝¹ : CategoryTheory.Preadditive C
                                                                          S : CategoryTheory.ShortComplex (CochainComplex C Int)
                                                                          σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
                                                                          inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
                                                                          K L : CochainComplex C Int
                                                                          φ : Quiver.Hom K L
                                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CochainComplex.mappingCone.triangle  …
                                                                        -/
  ShortComplex.mk (triangle φ).rotate.mor₁ (triangle φ).rotate.mor₂ (by simp)
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- `triangleRotateShortComplex φ` is a degreewise split short exact sequence of
cochain complexes. -/
@[simps]
noncomputable def triangleRotateShortComplexSplitting (n : ℤ) :
    ((triangleRotateShortComplex φ).map (eval _ _ n)).Splitting where
                                /-
                                  C : Type u_1
                                  inst✝² : CategoryTheory.Category.{?u.205836, u_1} C
                                  inst✝¹ : CategoryTheory.Preadditive C
                                  S : CategoryTheory.ShortComplex (CochainComplex C Int)
                                  σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
                                  inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
                                  K L : CochainComplex C Int
                                  φ : Quiver.Hom K L
                                  n : Int
                                  ⊢ Eq (HAdd.hAdd (HAdd.hAdd n 1) (-1)) n
                                -/
  s := -(inl φ).v (n + 1) n (by omega)
                                /-
                                  🎉 no goals
                                -/
  r := (snd φ).v n n (add_zero n)
           /-
             C : Type u_1
             inst✝² : CategoryTheory.Category.{?u.205836, u_1} C
             inst✝¹ : CategoryTheory.Preadditive C
             S : CategoryTheory.ShortComplex (CochainComplex C Int)
             σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
             inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
             K L : CochainComplex C Int
             φ : Quiver.Hom K L
             n : Int
             ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCo …
           -/
  id := by simp [ext_from_iff φ _ _ rfl]
           /-
             🎉 no goals
           -/


@[simp]
lemma cocycleOfDegreewiseSplit_triangleRotateShortComplexSplitting_v (p : ℤ) :
    (cocycleOfDegreewiseSplit _ (triangleRotateShortComplexSplitting φ)).1.v p _ rfl =
      -φ.f _ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    K L : CochainComplex C Int
    φ : Quiver.Hom K L
    p : Int
    ⊢ Eq ((↑(CochainComplex.cocycleOfDegreewiseSplit (CochainComplex.mappingCone.t …
  -/
  simp [cocycleOfDegreewiseSplit, d_snd_v φ p (p + 1) rfl]
  /-
    🎉 no goals
  -/


/-- The triangle `(triangle φ).rotate` is isomorphic to a triangle attached to a
degreewise split short exact sequence of cochain complexes. -/
noncomputable def triangleRotateIsoTriangleOfDegreewiseSplit :
    (triangle φ).rotate ≅
      triangleOfDegreewiseSplit _ (triangleRotateShortComplexSplitting φ) :=
  Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _) (Iso.refl _)
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.239117, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          S : CategoryTheory.ShortComplex (CochainComplex C Int)
          σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
          inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
          K L : CochainComplex C Int
          φ : Quiver.Hom K L
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CochainComplex.mappingCone.triangle  …
        -/
        /-
          🎉 no goals
        -/
                       /-
                         🎉 no goals
                       -/
    (by aesop_cat) (by aesop_cat) (by ext; dsimp; simp)
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- The triangle `(triangleh φ).rotate` is isomorphic to a triangle attached to a
degreewise split short exact sequence of cochain complexes. -/
noncomputable def trianglehRotateIsoTrianglehOfDegreewiseSplit :
    (triangleh φ).rotate ≅
      trianglehOfDegreewiseSplit _ (triangleRotateShortComplexSplitting φ) :=
  (HomotopyCategory.quotient _ _).mapTriangleRotateIso.app _ ≪≫
    (HomotopyCategory.quotient _ _).mapTriangle.mapIso
      (triangleRotateIsoTriangleOfDegreewiseSplit φ)


lemma distinguished_iff_iso_trianglehOfDegreewiseSplit
    (T : Triangle (HomotopyCategory C (ComplexShape.up ℤ))) :
    (T ∈ distTriang _) ↔ ∃ (S : ShortComplex (CochainComplex C ℤ))
      (σ : ∀ n, (S.map (HomologicalComplex.eval C _ n)).Splitting),
      Nonempty (T ≅ CochainComplex.trianglehOfDegreewiseSplit S σ) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    T : CategoryTheory.Pretriangulated.Triangle (HomotopyCategory C (ComplexShape. …
    ⊢ Iff (Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T) …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      T : CategoryTheory.Pretriangulated.Triangle (HomotopyCategory C (ComplexShape. …
      ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T → Exi …
    -/
  · intro hT
    /-
      case mp
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      T : CategoryTheory.Pretriangulated.Triangle (HomotopyCategory C (ComplexShape. …
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ Exists fun S => Exists fun σ => Nonempty (CategoryTheory.Iso T (CochainCompl …
    -/
    obtain ⟨K, L, φ, ⟨e⟩⟩ := inv_rot_of_distTriang _ hT
    exact ⟨_, _, ⟨(triangleRotation _).counitIso.symm.app _ ≪≫ (rotate _).mapIso e ≪≫
      CochainComplex.mappingCone.trianglehRotateIsoTrianglehOfDegreewiseSplit φ⟩⟩
    /-
      case mpr
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      T : CategoryTheory.Pretriangulated.Triangle (HomotopyCategory C (ComplexShape. …
      ⊢ (Exists fun S => Exists fun σ => Nonempty (CategoryTheory.Iso T (CochainComp …
    -/
  · rintro ⟨S, σ, ⟨e⟩⟩
    /-
      case mpr.intro.intro.intro
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      T : CategoryTheory.Pretriangulated.Triangle (HomotopyCategory C (ComplexShape. …
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
      e : CategoryTheory.Iso T (CochainComplex.trianglehOfDegreewiseSplit S σ)
      ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    -/
    rw [rotate_distinguished_triangle, rotate_distinguished_triangle]
    refine isomorphic_distinguished _ ?_ _
      ((rotate _ ⋙ rotate _).mapIso e ≪≫
        CochainComplex.trianglehOfDegreewiseSplitRotateRotateIso S σ)
    /-
      case mpr.intro.intro.intro
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
      T : CategoryTheory.Pretriangulated.Triangle (HomotopyCategory C (ComplexShape. …
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      σ : (n : Int) → (S.map (HomologicalComplex.eval C (ComplexShape.up Int) n)).Sp …
      e : CategoryTheory.Iso T (CochainComplex.trianglehOfDegreewiseSplit S σ)
      ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cochai …
    -/
    exact ⟨_, _, _, ⟨Iso.refl _⟩⟩
    /-
      🎉 no goals
    -/



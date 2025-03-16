local postfix:max "⋆" => star


lemma opNorm_mul_flip_apply (a : E) : ‖(mul 𝕜 E).flip a‖ = ‖a‖ := by
  refine le_antisymm
    (opNorm_le_bound _ (norm_nonneg _) fun b => by simpa only [mul_comm] using norm_mul_le b a) ?_
  suffices ‖mul 𝕜 E (star a)‖ ≤ ‖(mul 𝕜 E).flip a‖ by
    simpa only [ge_iff_le, opNorm_mul_apply, norm_star] using this
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NonUnitalNormedRing E
    inst✝⁵ : StarRing E
    inst✝⁴ : NormedStarGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : IsScalarTower 𝕜 E E
    inst✝¹ : SMulCommClass 𝕜 E E
    inst✝ : RegularNormedAlgebra 𝕜 E
    a : E
    ⊢ LE.le (Norm.norm ((ContinuousLinearMap.mul 𝕜 E) (Star.star a))) (Norm.norm ( …
  -/
  refine opNorm_le_bound _ (norm_nonneg _) fun b => ?_
  calc ‖mul 𝕜 E (star a) b‖ = ‖(mul 𝕜 E).flip a (star b)‖ := by
        simpa only [mul_apply', flip_apply, star_mul, star_star] using norm_star (star b * a)
    _ ≤ ‖(mul 𝕜 E).flip a‖ * ‖b‖ := by
        simpa only [flip_apply, mul_apply', norm_star] using le_opNorm ((mul 𝕜 E).flip a) (star b)


@[deprecated (since := "2024-02-02")] alias op_norm_mul_flip_apply := opNorm_mul_flip_apply


lemma opNNNorm_mul_flip_apply (a : E) : ‖(mul 𝕜 E).flip a‖₊ = ‖a‖₊ :=
  Subtype.ext (opNorm_mul_flip_apply 𝕜 a)


@[deprecated (since := "2024-02-02")] alias op_nnnorm_mul_flip_apply := opNNNorm_mul_flip_apply


lemma isometry_mul_flip : Isometry (mul 𝕜 E).flip :=
  AddMonoidHomClass.isometry_of_norm _ (opNorm_mul_flip_apply 𝕜)


/-- A C⋆-algebra over a densely normed field is a regular normed algebra. -/
instance CStarRing.instRegularNormedAlgebra : RegularNormedAlgebra 𝕜 E where
  isometry_mul' := AddMonoidHomClass.isometry_of_norm (mul 𝕜 E) fun a => NNReal.eq_iff.mp <|
    show ‖mul 𝕜 E a‖₊ = ‖a‖₊ by
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁶ : DenselyNormedField 𝕜
      inst✝⁵ : NonUnitalNormedRing E
      inst✝⁴ : StarRing E
      inst✝³ : CStarRing E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : IsScalarTower 𝕜 E E
      inst✝ : SMulCommClass 𝕜 E E
      a : E
      ⊢ Eq (NNNorm.nnnorm ((ContinuousLinearMap.mul 𝕜 E) a)) (NNNorm.nnnorm a)
    -/
    rw [← sSup_unitClosedBall_eq_nnnorm]
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁶ : DenselyNormedField 𝕜
      inst✝⁵ : NonUnitalNormedRing E
      inst✝⁴ : StarRing E
      inst✝³ : CStarRing E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : IsScalarTower 𝕜 E E
      inst✝ : SMulCommClass 𝕜 E E
      a : E
      ⊢ Eq (SupSet.sSup (Set.image (fun x => NNNorm.nnnorm (((ContinuousLinearMap.mu …
    -/
    refine csSup_eq_of_forall_le_of_forall_lt_exists_gt ?_ ?_ fun r hr => ?_
      /-
        case refine_1
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁶ : DenselyNormedField 𝕜
        inst✝⁵ : NonUnitalNormedRing E
        inst✝⁴ : StarRing E
        inst✝³ : CStarRing E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : IsScalarTower 𝕜 E E
        inst✝ : SMulCommClass 𝕜 E E
        a : E
        ⊢ (Set.image (fun x => NNNorm.nnnorm (((ContinuousLinearMap.mul 𝕜 E) a) x)) (M …
      -/
    · exact (Metric.nonempty_closedBall.mpr zero_le_one).image _
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁶ : DenselyNormedField 𝕜
        inst✝⁵ : NonUnitalNormedRing E
        inst✝⁴ : StarRing E
        inst✝³ : CStarRing E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : IsScalarTower 𝕜 E E
        inst✝ : SMulCommClass 𝕜 E E
        a : E
        ⊢ ∀ (a_1 : NNReal), Membership.mem (Set.image (fun x => NNNorm.nnnorm (((Conti …
      -/
    · rintro - ⟨x, hx, rfl⟩
      exact
        ((mul 𝕜 E a).unit_le_opNorm x <| mem_closedBall_zero_iff.mp hx).trans
          (opNorm_mul_apply_le 𝕜 E a)
      /-
        case refine_3
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁶ : DenselyNormedField 𝕜
        inst✝⁵ : NonUnitalNormedRing E
        inst✝⁴ : StarRing E
        inst✝³ : CStarRing E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : IsScalarTower 𝕜 E E
        inst✝ : SMulCommClass 𝕜 E E
        a : E
        r : NNReal
        hr : LT.lt r (NNNorm.nnnorm a)
        ⊢ Exists fun a_1 => And (Membership.mem (Set.image (fun x => NNNorm.nnnorm ((( …
      -/
    · have ha : 0 < ‖a‖₊ := zero_le'.trans_lt hr
      /-
        case refine_3
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁶ : DenselyNormedField 𝕜
        inst✝⁵ : NonUnitalNormedRing E
        inst✝⁴ : StarRing E
        inst✝³ : CStarRing E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : IsScalarTower 𝕜 E E
        inst✝ : SMulCommClass 𝕜 E E
        a : E
        r : NNReal
        hr : LT.lt r (NNNorm.nnnorm a)
        ha : LT.lt 0 (NNNorm.nnnorm a)
        ⊢ Exists fun a_1 => And (Membership.mem (Set.image (fun x => NNNorm.nnnorm ((( …
      -/
      rw [← inv_inv ‖a‖₊, NNReal.lt_inv_iff_mul_lt (inv_ne_zero ha.ne')] at hr
      obtain ⟨k, hk₁, hk₂⟩ :=
        NormedField.exists_lt_nnnorm_lt 𝕜 (mul_lt_mul_of_pos_right hr <| inv_pos.2 ha)
      /-
        case refine_3.intro.intro
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁶ : DenselyNormedField 𝕜
        inst✝⁵ : NonUnitalNormedRing E
        inst✝⁴ : StarRing E
        inst✝³ : CStarRing E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : IsScalarTower 𝕜 E E
        inst✝ : SMulCommClass 𝕜 E E
        a : E
        r : NNReal
        hr : LT.lt (HMul.hMul r (Inv.inv (NNNorm.nnnorm a))) 1
        ha : LT.lt 0 (NNNorm.nnnorm a)
        k : 𝕜
        hk₁ : LT.lt (HMul.hMul (HMul.hMul r (Inv.inv (NNNorm.nnnorm a))) (Inv.inv (NNN …
        hk₂ : LT.lt (NNNorm.nnnorm k) (HMul.hMul 1 (Inv.inv (NNNorm.nnnorm a)))
        ⊢ Exists fun a_1 => And (Membership.mem (Set.image (fun x => NNNorm.nnnorm ((( …
      -/
      refine ⟨_, ⟨k • star a, ?_, rfl⟩, ?_⟩
      · simpa only [mem_closedBall_zero_iff, norm_smul, one_mul, norm_star] using
          (NNReal.le_inv_iff_mul_le ha.ne').1 (one_mul ‖a‖₊⁻¹ ▸ hk₂.le : ‖k‖₊ ≤ ‖a‖₊⁻¹)
        /-
          case refine_3.intro.intro.refine_2
          𝕜 : Type u_1
          E : Type u_2
          inst✝⁶ : DenselyNormedField 𝕜
          inst✝⁵ : NonUnitalNormedRing E
          inst✝⁴ : StarRing E
          inst✝³ : CStarRing E
          inst✝² : NormedSpace 𝕜 E
          inst✝¹ : IsScalarTower 𝕜 E E
          inst✝ : SMulCommClass 𝕜 E E
          a : E
          r : NNReal
          hr : LT.lt (HMul.hMul r (Inv.inv (NNNorm.nnnorm a))) 1
          ha : LT.lt 0 (NNNorm.nnnorm a)
          k : 𝕜
          hk₁ : LT.lt (HMul.hMul (HMul.hMul r (Inv.inv (NNNorm.nnnorm a))) (Inv.inv (NNN …
          hk₂ : LT.lt (NNNorm.nnnorm k) (HMul.hMul 1 (Inv.inv (NNNorm.nnnorm a)))
          ⊢ LT.lt r ((fun x => NNNorm.nnnorm (((ContinuousLinearMap.mul 𝕜 E) a) x)) (HSM …
        -/
      · simp only [map_smul, nnnorm_smul, mul_apply', mul_smul_comm, CStarRing.nnnorm_self_mul_star]
        /-
          case refine_3.intro.intro.refine_2
          𝕜 : Type u_1
          E : Type u_2
          inst✝⁶ : DenselyNormedField 𝕜
          inst✝⁵ : NonUnitalNormedRing E
          inst✝⁴ : StarRing E
          inst✝³ : CStarRing E
          inst✝² : NormedSpace 𝕜 E
          inst✝¹ : IsScalarTower 𝕜 E E
          inst✝ : SMulCommClass 𝕜 E E
          a : E
          r : NNReal
          hr : LT.lt (HMul.hMul r (Inv.inv (NNNorm.nnnorm a))) 1
          ha : LT.lt 0 (NNNorm.nnnorm a)
          k : 𝕜
          hk₁ : LT.lt (HMul.hMul (HMul.hMul r (Inv.inv (NNNorm.nnnorm a))) (Inv.inv (NNN …
          hk₂ : LT.lt (NNNorm.nnnorm k) (HMul.hMul 1 (Inv.inv (NNNorm.nnnorm a)))
          ⊢ LT.lt r (HMul.hMul (NNNorm.nnnorm k) (HMul.hMul (NNNorm.nnnorm a) (NNNorm.nn …
        -/
        rwa [← div_lt_iff₀ (mul_pos ha ha), div_eq_mul_inv, mul_inv, ← mul_assoc]
        /-
          🎉 no goals
        -/


/-- This is the key lemma used to establish the instance `Unitization.instCStarRing`
(i.e., proving that the norm on `Unitization 𝕜 E` satisfies the C⋆-property). We split this one
out so that declaring the `CStarRing` instance doesn't time out. -/
theorem Unitization.norm_splitMul_snd_sq (x : Unitization 𝕜 E) :
    ‖(Unitization.splitMul 𝕜 E x).snd‖ ^ 2 ≤ ‖(Unitization.splitMul 𝕜 E (star x * x)).snd‖ := by
  /- The key idea is that we can use `sSup_unitClosedBall_eq_norm` to make this about
  applying this linear map to elements of norm at most one. There is a bit of `sqrt` and `sq`
  shuffling that needs to occur, which is primarily just an annoyance. -/
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : DenselyNormedField 𝕜
    inst✝⁷ : NonUnitalNormedRing E
    inst✝⁶ : StarRing E
    inst✝⁵ : CStarRing E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : IsScalarTower 𝕜 E E
    inst✝² : SMulCommClass 𝕜 E E
    inst✝¹ : StarRing 𝕜
    inst✝ : StarModule 𝕜 E
    x : Unitization 𝕜 E
    ⊢ LE.le (HPow.hPow (Norm.norm ((Unitization.splitMul 𝕜 E) x).2) 2) (Norm.norm  …
  -/
  refine (Real.le_sqrt (norm_nonneg _) (norm_nonneg _)).mp ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : DenselyNormedField 𝕜
    inst✝⁷ : NonUnitalNormedRing E
    inst✝⁶ : StarRing E
    inst✝⁵ : CStarRing E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : IsScalarTower 𝕜 E E
    inst✝² : SMulCommClass 𝕜 E E
    inst✝¹ : StarRing 𝕜
    inst✝ : StarModule 𝕜 E
    x : Unitization 𝕜 E
    ⊢ LE.le (Norm.norm ((Unitization.splitMul 𝕜 E) x).2) (Norm.norm ((Unitization. …
  -/
  simp only [Unitization.splitMul_apply]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : DenselyNormedField 𝕜
    inst✝⁷ : NonUnitalNormedRing E
    inst✝⁶ : StarRing E
    inst✝⁵ : CStarRing E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : IsScalarTower 𝕜 E E
    inst✝² : SMulCommClass 𝕜 E E
    inst✝¹ : StarRing 𝕜
    inst✝ : StarModule 𝕜 E
    x : Unitization 𝕜 E
    ⊢ LE.le (Norm.norm (HAdd.hAdd ((algebraMap 𝕜 (ContinuousLinearMap (RingHom.id  …
  -/
  rw [← sSup_unitClosedBall_eq_norm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : DenselyNormedField 𝕜
    inst✝⁷ : NonUnitalNormedRing E
    inst✝⁶ : StarRing E
    inst✝⁵ : CStarRing E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : IsScalarTower 𝕜 E E
    inst✝² : SMulCommClass 𝕜 E E
    inst✝¹ : StarRing 𝕜
    inst✝ : StarModule 𝕜 E
    x : Unitization 𝕜 E
    ⊢ LE.le (SupSet.sSup (Set.image (fun x_1 => Norm.norm ((HAdd.hAdd ((algebraMap …
  -/
  refine csSup_le ((Metric.nonempty_closedBall.2 zero_le_one).image _) ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : DenselyNormedField 𝕜
    inst✝⁷ : NonUnitalNormedRing E
    inst✝⁶ : StarRing E
    inst✝⁵ : CStarRing E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : IsScalarTower 𝕜 E E
    inst✝² : SMulCommClass 𝕜 E E
    inst✝¹ : StarRing 𝕜
    inst✝ : StarModule 𝕜 E
    x : Unitization 𝕜 E
    ⊢ ∀ (b : Real), Membership.mem (Set.image (fun x_1 => Norm.norm ((HAdd.hAdd (( …
  -/
  rintro - ⟨b, hb, rfl⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : DenselyNormedField 𝕜
    inst✝⁷ : NonUnitalNormedRing E
    inst✝⁶ : StarRing E
    inst✝⁵ : CStarRing E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : IsScalarTower 𝕜 E E
    inst✝² : SMulCommClass 𝕜 E E
    inst✝¹ : StarRing 𝕜
    inst✝ : StarModule 𝕜 E
    x : Unitization 𝕜 E
    b : E
    hb : Membership.mem (Metric.closedBall 0 1) b
    ⊢ LE.le ((fun x_1 => Norm.norm ((HAdd.hAdd ((algebraMap 𝕜 (ContinuousLinearMap …
  -/
  simp only
  -- rewrite to a more convenient form; this is where we use the C⋆-property
  rw [← Real.sqrt_sq (norm_nonneg _), Real.sqrt_le_sqrt_iff (norm_nonneg _), sq,
    ← CStarRing.norm_star_mul_self, ContinuousLinearMap.add_apply, star_add, mul_apply',
    Algebra.algebraMap_eq_smul_one, ContinuousLinearMap.smul_apply,
    ContinuousLinearMap.one_apply, star_mul, star_smul, add_mul, smul_mul_assoc, ← mul_smul_comm,
    mul_assoc, ← mul_add, ← sSup_unitClosedBall_eq_norm]
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : DenselyNormedField 𝕜
    inst✝⁷ : NonUnitalNormedRing E
    inst✝⁶ : StarRing E
    inst✝⁵ : CStarRing E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : IsScalarTower 𝕜 E E
    inst✝² : SMulCommClass 𝕜 E E
    inst✝¹ : StarRing 𝕜
    inst✝ : StarModule 𝕜 E
    x : Unitization 𝕜 E
    b : E
    hb : Membership.mem (Metric.closedBall 0 1) b
    ⊢ LE.le (Norm.norm (HMul.hMul (Star.star b) (HAdd.hAdd (HSMul.hSMul (Star.star …
  -/
  refine (norm_mul_le _ _).trans ?_
  calc
    _ ≤ ‖star x.fst • (x.fst • b + x.snd * b) + star x.snd * (x.fst • b + x.snd * b)‖ := by
      nth_rewrite 2 [← one_mul ‖_ + _‖]
      gcongr
      exact (norm_star b).symm ▸ mem_closedBall_zero_iff.1 hb
    _ ≤ sSup (_ '' Metric.closedBall 0 1) := le_csSup ?_ ⟨b, hb, ?_⟩
  -- now we just check the side conditions for `le_csSup`. There is nothing of interest here.
    /-
      case intro.intro.calc_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁸ : DenselyNormedField 𝕜
      inst✝⁷ : NonUnitalNormedRing E
      inst✝⁶ : StarRing E
      inst✝⁵ : CStarRing E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : IsScalarTower 𝕜 E E
      inst✝² : SMulCommClass 𝕜 E E
      inst✝¹ : StarRing 𝕜
      inst✝ : StarModule 𝕜 E
      x : Unitization 𝕜 E
      b : E
      hb : Membership.mem (Metric.closedBall 0 1) b
      ⊢ BddAbove (Set.image (fun x_1 => Norm.norm ((HAdd.hAdd ((algebraMap 𝕜 (Contin …
    -/
  · refine ⟨‖(star x * x).fst‖ + ‖(star x * x).snd‖, ?_⟩
    /-
      case intro.intro.calc_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁸ : DenselyNormedField 𝕜
      inst✝⁷ : NonUnitalNormedRing E
      inst✝⁶ : StarRing E
      inst✝⁵ : CStarRing E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : IsScalarTower 𝕜 E E
      inst✝² : SMulCommClass 𝕜 E E
      inst✝¹ : StarRing 𝕜
      inst✝ : StarModule 𝕜 E
      x : Unitization 𝕜 E
      b : E
      hb : Membership.mem (Metric.closedBall 0 1) b
      ⊢ Membership.mem (upperBounds (Set.image (fun x_1 => Norm.norm ((HAdd.hAdd ((a …
    -/
    rintro _ ⟨y, hy, rfl⟩
    /-
      case intro.intro.calc_1.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁸ : DenselyNormedField 𝕜
      inst✝⁷ : NonUnitalNormedRing E
      inst✝⁶ : StarRing E
      inst✝⁵ : CStarRing E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : IsScalarTower 𝕜 E E
      inst✝² : SMulCommClass 𝕜 E E
      inst✝¹ : StarRing 𝕜
      inst✝ : StarModule 𝕜 E
      x : Unitization 𝕜 E
      b : E
      hb : Membership.mem (Metric.closedBall 0 1) b
      y : E
      hy : Membership.mem (Metric.closedBall 0 1) y
      ⊢ LE.le ((fun x_1 => Norm.norm ((HAdd.hAdd ((algebraMap 𝕜 (ContinuousLinearMap …
    -/
    refine (norm_add_le _ _).trans ?_
    /-
      case intro.intro.calc_1.intro.intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁸ : DenselyNormedField 𝕜
      inst✝⁷ : NonUnitalNormedRing E
      inst✝⁶ : StarRing E
      inst✝⁵ : CStarRing E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : IsScalarTower 𝕜 E E
      inst✝² : SMulCommClass 𝕜 E E
      inst✝¹ : StarRing 𝕜
      inst✝ : StarModule 𝕜 E
      x : Unitization 𝕜 E
      b : E
      hb : Membership.mem (Metric.closedBall 0 1) b
      y : E
      hy : Membership.mem (Metric.closedBall 0 1) y
      ⊢ LE.le (HAdd.hAdd (Norm.norm (↑((algebraMap 𝕜 (ContinuousLinearMap (RingHom.i …
    -/
    gcongr
      /-
        case intro.intro.calc_1.intro.intro.h₁
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁸ : DenselyNormedField 𝕜
        inst✝⁷ : NonUnitalNormedRing E
        inst✝⁶ : StarRing E
        inst✝⁵ : CStarRing E
        inst✝⁴ : NormedSpace 𝕜 E
        inst✝³ : IsScalarTower 𝕜 E E
        inst✝² : SMulCommClass 𝕜 E E
        inst✝¹ : StarRing 𝕜
        inst✝ : StarModule 𝕜 E
        x : Unitization 𝕜 E
        b : E
        hb : Membership.mem (Metric.closedBall 0 1) b
        y : E
        hy : Membership.mem (Metric.closedBall 0 1) y
        ⊢ LE.le (Norm.norm (↑((algebraMap 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) E E))  …
      -/
    · rw [Algebra.algebraMap_eq_smul_one]
      /-
        case intro.intro.calc_1.intro.intro.h₁
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁸ : DenselyNormedField 𝕜
        inst✝⁷ : NonUnitalNormedRing E
        inst✝⁶ : StarRing E
        inst✝⁵ : CStarRing E
        inst✝⁴ : NormedSpace 𝕜 E
        inst✝³ : IsScalarTower 𝕜 E E
        inst✝² : SMulCommClass 𝕜 E E
        inst✝¹ : StarRing 𝕜
        inst✝ : StarModule 𝕜 E
        x : Unitization 𝕜 E
        b : E
        hb : Membership.mem (Metric.closedBall 0 1) b
        y : E
        hy : Membership.mem (Metric.closedBall 0 1) y
        ⊢ LE.le (Norm.norm (↑(HSMul.hSMul (HMul.hMul (Star.star x) x).fst 1) y)) (Norm …
      -/
      refine (norm_smul _ _).trans_le ?_
      simpa only [mul_one] using
        mul_le_mul_of_nonneg_left (mem_closedBall_zero_iff.1 hy) (norm_nonneg (star x * x).fst)
      /-
        case intro.intro.calc_1.intro.intro.h₂
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁸ : DenselyNormedField 𝕜
        inst✝⁷ : NonUnitalNormedRing E
        inst✝⁶ : StarRing E
        inst✝⁵ : CStarRing E
        inst✝⁴ : NormedSpace 𝕜 E
        inst✝³ : IsScalarTower 𝕜 E E
        inst✝² : SMulCommClass 𝕜 E E
        inst✝¹ : StarRing 𝕜
        inst✝ : StarModule 𝕜 E
        x : Unitization 𝕜 E
        b : E
        hb : Membership.mem (Metric.closedBall 0 1) b
        y : E
        hy : Membership.mem (Metric.closedBall 0 1) y
        ⊢ LE.le (Norm.norm (↑((ContinuousLinearMap.mul 𝕜 E) (HMul.hMul (Star.star x) x …
      -/
    · exact (unit_le_opNorm _ y <| mem_closedBall_zero_iff.1 hy).trans (opNorm_mul_apply_le _ _ _)
      /-
        🎉 no goals
      -/
  · simp only [ContinuousLinearMap.add_apply, mul_apply', Unitization.snd_star, Unitization.snd_mul,
      Unitization.fst_mul, Unitization.fst_star, Algebra.algebraMap_eq_smul_one, smul_apply,
      one_apply, smul_add, mul_add, add_mul]
    /-
      case intro.intro.calc_2
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁸ : DenselyNormedField 𝕜
      inst✝⁷ : NonUnitalNormedRing E
      inst✝⁶ : StarRing E
      inst✝⁵ : CStarRing E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : IsScalarTower 𝕜 E E
      inst✝² : SMulCommClass 𝕜 E E
      inst✝¹ : StarRing 𝕜
      inst✝ : StarModule 𝕜 E
      x : Unitization 𝕜 E
      b : E
      hb : Membership.mem (Metric.closedBall 0 1) b
      ⊢ Eq (Norm.norm (HAdd.hAdd (HSMul.hSMul (HMul.hMul (Star.star x.fst) x.fst) b) …
    -/
    simp only [smul_smul, smul_mul_assoc, ← add_assoc, ← mul_assoc, mul_smul_comm]
    /-
      🎉 no goals
    -/


/-- The norm on `Unitization 𝕜 E` satisfies the C⋆-property -/
instance Unitization.instCStarRing : CStarRing (Unitization 𝕜 E) where
  norm_mul_self_le x := by
    -- rewrite both sides as a `⊔`
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁹ : DenselyNormedField 𝕜
      inst✝⁸ : NonUnitalNormedRing E
      inst✝⁷ : StarRing E
      inst✝⁶ : CStarRing E
      inst✝⁵ : NormedSpace 𝕜 E
      inst✝⁴ : IsScalarTower 𝕜 E E
      inst✝³ : SMulCommClass 𝕜 E E
      inst✝² : StarRing 𝕜
      inst✝¹ : StarModule 𝕜 E
      inst✝ : CStarRing 𝕜
      x : Unitization 𝕜 E
      ⊢ LE.le (HMul.hMul (Norm.norm x) (Norm.norm x)) (Norm.norm (HMul.hMul (Star.st …
    -/
    simp only [Unitization.norm_def, Prod.norm_def]
    -- Show that `(Unitization.splitMul 𝕜 E x).snd` satisfies the C⋆-property, in two stages:
    have h₁ : ∀ x : Unitization 𝕜 E,
        ‖(Unitization.splitMul 𝕜 E x).snd‖ ≤ ‖(Unitization.splitMul 𝕜 E (star x)).snd‖ := by
      simp only [add_zero, Unitization.splitMul_apply, Unitization.snd_star, Unitization.fst_star]
      intro x
      /- split based on whether the term inside the norm is zero or not. If so, it's trivial.
      If not, then apply `norm_splitMul_snd_sq` and cancel one copy of the norm -/
      by_cases h : algebraMap 𝕜 (E →L[𝕜] E) x.fst + mul 𝕜 E x.snd = 0
      · simp only [h, norm_zero, norm_le_zero_iff]
        exact norm_nonneg _
      · have : ‖(Unitization.splitMul 𝕜 E x).snd‖ ^ 2 ≤
          ‖(Unitization.splitMul 𝕜 E (star x)).snd‖ * ‖(Unitization.splitMul 𝕜 E x).snd‖ :=
          (norm_splitMul_snd_sq 𝕜 x).trans <| by
            rw [map_mul, Prod.snd_mul]
            exact norm_mul_le _ _
        rw [sq] at this
        rw [← Ne, ← norm_pos_iff] at h
        simp only [add_zero, Unitization.splitMul_apply, Unitization.snd_star,
          Unitization.fst_star, star_star] at this
        exact (mul_le_mul_right h).mp this
    -- in this step we make use of the key lemma `norm_splitMul_snd_sq`
    have h₂ : ‖(Unitization.splitMul 𝕜 E (star x * x)).snd‖
        = ‖(Unitization.splitMul 𝕜 E x).snd‖ ^ 2 := by
      refine le_antisymm ?_ (norm_splitMul_snd_sq 𝕜 x)
      rw [map_mul, Prod.snd_mul]
      exact (norm_mul_le _ _).trans <| by
        rw [sq]
        gcongr
        simpa only [star_star] using h₁ (star x)
    -- Show that `(Unitization.splitMul 𝕜 E x).fst` satisfies the C⋆-property
    have h₃ : ‖(Unitization.splitMul 𝕜 E (star x * x)).fst‖
        = ‖(Unitization.splitMul 𝕜 E x).fst‖ ^ 2 := by
      simp only [Unitization.splitMul_apply, Unitization.fst_mul, Unitization.fst_star, add_zero,
        norm_mul, norm_star, sq]
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁹ : DenselyNormedField 𝕜
      inst✝⁸ : NonUnitalNormedRing E
      inst✝⁷ : StarRing E
      inst✝⁶ : CStarRing E
      inst✝⁵ : NormedSpace 𝕜 E
      inst✝⁴ : IsScalarTower 𝕜 E E
      inst✝³ : SMulCommClass 𝕜 E E
      inst✝² : StarRing 𝕜
      inst✝¹ : StarModule 𝕜 E
      inst✝ : CStarRing 𝕜
      x : Unitization 𝕜 E
      h₁ : ∀ (x : Unitization 𝕜 E), LE.le (Norm.norm ((Unitization.splitMul 𝕜 E) x). …
      h₂ : Eq (Norm.norm ((Unitization.splitMul 𝕜 E) (HMul.hMul (Star.star x) x)).2) …
      h₃ : Eq (Norm.norm ((Unitization.splitMul 𝕜 E) (HMul.hMul (Star.star x) x)).1) …
      ⊢ LE.le (HMul.hMul (Max.max (Norm.norm ((Unitization.splitMul 𝕜 E) x).1) (Norm …
    -/
    rw [h₂, h₃]
    /- use the definition of the norm, and split into cases based on whether the norm in the first
    coordinate is bigger or smaller than the norm in the second coordinate. -/
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁹ : DenselyNormedField 𝕜
      inst✝⁸ : NonUnitalNormedRing E
      inst✝⁷ : StarRing E
      inst✝⁶ : CStarRing E
      inst✝⁵ : NormedSpace 𝕜 E
      inst✝⁴ : IsScalarTower 𝕜 E E
      inst✝³ : SMulCommClass 𝕜 E E
      inst✝² : StarRing 𝕜
      inst✝¹ : StarModule 𝕜 E
      inst✝ : CStarRing 𝕜
      x : Unitization 𝕜 E
      h₁ : ∀ (x : Unitization 𝕜 E), LE.le (Norm.norm ((Unitization.splitMul 𝕜 E) x). …
      h₂ : Eq (Norm.norm ((Unitization.splitMul 𝕜 E) (HMul.hMul (Star.star x) x)).2) …
      h₃ : Eq (Norm.norm ((Unitization.splitMul 𝕜 E) (HMul.hMul (Star.star x) x)).1) …
      ⊢ LE.le (HMul.hMul (Max.max (Norm.norm ((Unitization.splitMul 𝕜 E) x).1) (Norm …
    -/
    by_cases h : ‖(Unitization.splitMul 𝕜 E x).fst‖ ≤ ‖(Unitization.splitMul 𝕜 E x).snd‖
      /-
        case pos
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁹ : DenselyNormedField 𝕜
        inst✝⁸ : NonUnitalNormedRing E
        inst✝⁷ : StarRing E
        inst✝⁶ : CStarRing E
        inst✝⁵ : NormedSpace 𝕜 E
        inst✝⁴ : IsScalarTower 𝕜 E E
        inst✝³ : SMulCommClass 𝕜 E E
        inst✝² : StarRing 𝕜
        inst✝¹ : StarModule 𝕜 E
        inst✝ : CStarRing 𝕜
        x : Unitization 𝕜 E
        h₁ : ∀ (x : Unitization 𝕜 E), LE.le (Norm.norm ((Unitization.splitMul 𝕜 E) x). …
        h₂ : Eq (Norm.norm ((Unitization.splitMul 𝕜 E) (HMul.hMul (Star.star x) x)).2) …
        h₃ : Eq (Norm.norm ((Unitization.splitMul 𝕜 E) (HMul.hMul (Star.star x) x)).1) …
        h : LE.le (Norm.norm ((Unitization.splitMul 𝕜 E) x).1) (Norm.norm ((Unitizatio …
        ⊢ LE.le (HMul.hMul (Max.max (Norm.norm ((Unitization.splitMul 𝕜 E) x).1) (Norm …
      -/
    · rw [sq, sq, sup_eq_right.mpr h, sup_eq_right.mpr (mul_self_le_mul_self (norm_nonneg _) h)]
      /-
        🎉 no goals
      -/
      /-
        case neg
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁹ : DenselyNormedField 𝕜
        inst✝⁸ : NonUnitalNormedRing E
        inst✝⁷ : StarRing E
        inst✝⁶ : CStarRing E
        inst✝⁵ : NormedSpace 𝕜 E
        inst✝⁴ : IsScalarTower 𝕜 E E
        inst✝³ : SMulCommClass 𝕜 E E
        inst✝² : StarRing 𝕜
        inst✝¹ : StarModule 𝕜 E
        inst✝ : CStarRing 𝕜
        x : Unitization 𝕜 E
        h₁ : ∀ (x : Unitization 𝕜 E), LE.le (Norm.norm ((Unitization.splitMul 𝕜 E) x). …
        h₂ : Eq (Norm.norm ((Unitization.splitMul 𝕜 E) (HMul.hMul (Star.star x) x)).2) …
        h₃ : Eq (Norm.norm ((Unitization.splitMul 𝕜 E) (HMul.hMul (Star.star x) x)).1) …
        h : Not (LE.le (Norm.norm ((Unitization.splitMul 𝕜 E) x).1) (Norm.norm ((Uniti …
        ⊢ LE.le (HMul.hMul (Max.max (Norm.norm ((Unitization.splitMul 𝕜 E) x).1) (Norm …
      -/
    · replace h := (not_le.mp h).le
      /-
        case neg
        𝕜 : Type u_1
        E : Type u_2
        inst✝⁹ : DenselyNormedField 𝕜
        inst✝⁸ : NonUnitalNormedRing E
        inst✝⁷ : StarRing E
        inst✝⁶ : CStarRing E
        inst✝⁵ : NormedSpace 𝕜 E
        inst✝⁴ : IsScalarTower 𝕜 E E
        inst✝³ : SMulCommClass 𝕜 E E
        inst✝² : StarRing 𝕜
        inst✝¹ : StarModule 𝕜 E
        inst✝ : CStarRing 𝕜
        x : Unitization 𝕜 E
        h₁ : ∀ (x : Unitization 𝕜 E), LE.le (Norm.norm ((Unitization.splitMul 𝕜 E) x). …
        h₂ : Eq (Norm.norm ((Unitization.splitMul 𝕜 E) (HMul.hMul (Star.star x) x)).2) …
        h₃ : Eq (Norm.norm ((Unitization.splitMul 𝕜 E) (HMul.hMul (Star.star x) x)).1) …
        h : LE.le (Norm.norm ((Unitization.splitMul 𝕜 E) x).2) (Norm.norm ((Unitizatio …
        ⊢ LE.le (HMul.hMul (Max.max (Norm.norm ((Unitization.splitMul 𝕜 E) x).1) (Norm …
      -/
      rw [sq, sq, sup_eq_left.mpr h, sup_eq_left.mpr (mul_self_le_mul_self (norm_nonneg _) h)]
      /-
        🎉 no goals
      -/


/-- The minimal unitization (over `ℂ`) of a C⋆-algebra, equipped with the C⋆-norm. When `A` is
unital, `A⁺¹ ≃⋆ₐ[ℂ] (ℂ × A)`. -/
scoped[CStarAlgebra] postfix:max "⁺¹" => Unitization ℂ


noncomputable instance Unitization.instCStarAlgebra {A : Type*} [NonUnitalCStarAlgebra A] :
    CStarAlgebra (Unitization ℂ A) where


noncomputable instance Unitization.instCommCStarAlgebra {A : Type*} [NonUnitalCommCStarAlgebra A] :
    CommCStarAlgebra (Unitization ℂ A) where
  mul_comm := mul_comm



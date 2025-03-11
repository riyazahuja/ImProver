/-- The *spectral radius* is the supremum of the `nnnorm` (`‖·‖₊`) of elements in the spectrum,
    coerced into an element of `ℝ≥0∞`. Note that it is possible for `spectrum 𝕜 a = ∅`. In this
    case, `spectralRadius a = 0`. It is also possible that `spectrum 𝕜 a` be unbounded (though
    not for Banach algebras, see `spectrum.isBounded`, below).  In this case,
    `spectralRadius a = ∞`. -/
noncomputable def spectralRadius (𝕜 : Type*) {A : Type*} [NormedField 𝕜] [Ring A] [Algebra 𝕜 A]
    (a : A) : ℝ≥0∞ :=
  ⨆ k ∈ spectrum 𝕜 a, ‖k‖₊


local notation "σ" => spectrum 𝕜

local notation "ρ" => resolventSet 𝕜

local notation "↑ₐ" => algebraMap 𝕜 A


@[simp]
theorem SpectralRadius.of_subsingleton [Subsingleton A] (a : A) : spectralRadius 𝕜 a = 0 := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : Subsingleton A
    a : A
    ⊢ Eq (spectralRadius 𝕜 a) 0
  -/
  simp [spectralRadius]
  /-
    🎉 no goals
  -/


@[simp]
theorem spectralRadius_zero : spectralRadius 𝕜 (0 : A) = 0 := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedRing A
    inst✝ : NormedAlgebra 𝕜 A
    ⊢ Eq (spectralRadius 𝕜 0) 0
  -/
  nontriviality A
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedRing A
    inst✝ : NormedAlgebra 𝕜 A
    a✝ : Nontrivial A
    ⊢ Eq (spectralRadius 𝕜 0) 0
  -/
  simp [spectralRadius]
  /-
    🎉 no goals
  -/


theorem mem_resolventSet_of_spectralRadius_lt {a : A} {k : 𝕜} (h : spectralRadius 𝕜 a < ‖k‖₊) :
    k ∈ ρ a :=
  Classical.not_not.mp fun hn => h.not_le <| le_iSup₂ (α := ℝ≥0∞) k hn


theorem isOpen_resolventSet (a : A) : IsOpen (ρ a) :=
  Units.isOpen.preimage ((continuous_algebraMap 𝕜 A).sub continuous_const)


protected theorem isClosed (a : A) : IsClosed (σ a) :=
  (isOpen_resolventSet a).isClosed_compl


theorem mem_resolventSet_of_norm_lt_mul {a : A} {k : 𝕜} (h : ‖a‖ * ‖(1 : A)‖ < ‖k‖) : k ∈ ρ a := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    k : 𝕜
    h : LT.lt (HMul.hMul (Norm.norm a) (Norm.norm 1)) (Norm.norm k)
    ⊢ Membership.mem (resolventSet 𝕜 a) k
  -/
  rw [resolventSet, Set.mem_setOf_eq, Algebra.algebraMap_eq_smul_one]
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    k : 𝕜
    h : LT.lt (HMul.hMul (Norm.norm a) (Norm.norm 1)) (Norm.norm k)
    ⊢ IsUnit (HSub.hSub (HSMul.hSMul k 1) a)
  -/
  nontriviality A
  have hk : k ≠ 0 :=
    ne_zero_of_norm_ne_zero ((mul_nonneg (norm_nonneg _) (norm_nonneg _)).trans_lt h).ne'
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    k : 𝕜
    h : LT.lt (HMul.hMul (Norm.norm a) (Norm.norm 1)) (Norm.norm k)
    a✝ : Nontrivial A
    hk : Ne k 0
    ⊢ IsUnit (HSub.hSub (HSMul.hSMul k 1) a)
  -/
  letI ku := Units.map ↑ₐ.toMonoidHom (Units.mk0 k hk)
  rw [← inv_inv ‖(1 : A)‖,
    mul_inv_lt_iff₀' (inv_pos.2 <| norm_pos_iff.2 (one_ne_zero : (1 : A) ≠ 0))] at h
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    k : 𝕜
    h : LT.lt (Norm.norm a) (HMul.hMul (Inv.inv (Norm.norm 1)) (Norm.norm k))
    a✝ : Nontrivial A
    hk : Ne k 0
    ku : Units A := (Units.map ↑(algebraMap 𝕜 A)) (Units.mk0 k hk)
    ⊢ IsUnit (HSub.hSub (HSMul.hSMul k 1) a)
  -/
  have hku : ‖-a‖ < ‖(↑ku⁻¹ : A)‖⁻¹ := by simpa [ku, norm_algebraMap] using h
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    k : 𝕜
    h : LT.lt (Norm.norm a) (HMul.hMul (Inv.inv (Norm.norm 1)) (Norm.norm k))
    a✝ : Nontrivial A
    hk : Ne k 0
    ku : Units A := (Units.map ↑(algebraMap 𝕜 A)) (Units.mk0 k hk)
    hku : LT.lt (Norm.norm (Neg.neg a)) (Inv.inv (Norm.norm ↑(Inv.inv ku)))
    ⊢ IsUnit (HSub.hSub (HSMul.hSMul k 1) a)
  -/
  simpa [ku, sub_eq_add_neg, Algebra.algebraMap_eq_smul_one] using (ku.add (-a) hku).isUnit
  /-
    🎉 no goals
  -/


theorem mem_resolventSet_of_norm_lt [NormOneClass A] {a : A} {k : 𝕜} (h : ‖a‖ < ‖k‖) : k ∈ ρ a :=
                                      /-
                                        𝕜 : Type u_1
                                        A : Type u_2
                                        inst✝⁴ : NormedField 𝕜
                                        inst✝³ : NormedRing A
                                        inst✝² : NormedAlgebra 𝕜 A
                                        inst✝¹ : CompleteSpace A
                                        inst✝ : NormOneClass A
                                        a : A
                                        k : 𝕜
                                        h : LT.lt (Norm.norm a) (Norm.norm k)
                                        ⊢ LT.lt (HMul.hMul (Norm.norm a) (Norm.norm 1)) (Norm.norm k)
                                      -/
  mem_resolventSet_of_norm_lt_mul (by rwa [norm_one, mul_one])
                                      /-
                                        🎉 no goals
                                      -/


theorem norm_le_norm_mul_of_mem {a : A} {k : 𝕜} (hk : k ∈ σ a) : ‖k‖ ≤ ‖a‖ * ‖(1 : A)‖ :=
  le_of_not_lt <| mt mem_resolventSet_of_norm_lt_mul hk


theorem norm_le_norm_of_mem [NormOneClass A] {a : A} {k : 𝕜} (hk : k ∈ σ a) : ‖k‖ ≤ ‖a‖ :=
  le_of_not_lt <| mt mem_resolventSet_of_norm_lt hk


theorem subset_closedBall_norm_mul (a : A) : σ a ⊆ Metric.closedBall (0 : 𝕜) (‖a‖ * ‖(1 : A)‖) :=
                 /-
                   𝕜 : Type u_1
                   A : Type u_2
                   inst✝³ : NormedField 𝕜
                   inst✝² : NormedRing A
                   inst✝¹ : NormedAlgebra 𝕜 A
                   inst✝ : CompleteSpace A
                   a : A
                   k : 𝕜
                   hk : Membership.mem (spectrum 𝕜 a) k
                   ⊢ Membership.mem (Metric.closedBall 0 (HMul.hMul (Norm.norm a) (Norm.norm 1))) k
                 -/
  fun k hk => by simp [norm_le_norm_mul_of_mem hk]
                 /-
                   🎉 no goals
                 -/


theorem subset_closedBall_norm [NormOneClass A] (a : A) : σ a ⊆ Metric.closedBall (0 : 𝕜) ‖a‖ :=
                 /-
                   𝕜 : Type u_1
                   A : Type u_2
                   inst✝⁴ : NormedField 𝕜
                   inst✝³ : NormedRing A
                   inst✝² : NormedAlgebra 𝕜 A
                   inst✝¹ : CompleteSpace A
                   inst✝ : NormOneClass A
                   a : A
                   k : 𝕜
                   hk : Membership.mem (spectrum 𝕜 a) k
                   ⊢ Membership.mem (Metric.closedBall 0 (Norm.norm a)) k
                 -/
  fun k hk => by simp [norm_le_norm_of_mem hk]
                 /-
                   🎉 no goals
                 -/


theorem isBounded (a : A) : Bornology.IsBounded (σ a) :=
  Metric.isBounded_closedBall.subset (subset_closedBall_norm_mul a)


protected theorem isCompact [ProperSpace 𝕜] (a : A) : IsCompact (σ a) :=
  Metric.isCompact_of_isClosed_isBounded (spectrum.isClosed a) (isBounded a)


instance instCompactSpace [ProperSpace 𝕜] (a : A) : CompactSpace (spectrum 𝕜 a) :=
  isCompact_iff_compactSpace.mp <| spectrum.isCompact a


instance instCompactSpaceNNReal {A : Type*} [NormedRing A] [NormedAlgebra ℝ A]
    (a : A) [CompactSpace (spectrum ℝ a)] : CompactSpace (spectrum ℝ≥0 a) := by
  /-
    𝕜 : Type u_1
    A✝ : Type u_2
    inst✝⁶ : NormedField 𝕜
    inst✝⁵ : NormedRing A✝
    inst✝⁴ : NormedAlgebra 𝕜 A✝
    inst✝³ : CompleteSpace A✝
    A : Type u_3
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    a : A
    inst✝ : CompactSpace ↑(spectrum Real a)
    ⊢ CompactSpace ↑(spectrum NNReal a)
  -/
  rw [← isCompact_iff_compactSpace] at *
  /-
    𝕜 : Type u_1
    A✝ : Type u_2
    inst✝⁶ : NormedField 𝕜
    inst✝⁵ : NormedRing A✝
    inst✝⁴ : NormedAlgebra 𝕜 A✝
    inst✝³ : CompleteSpace A✝
    A : Type u_3
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    a : A
    inst✝ : IsCompact (spectrum Real a)
    ⊢ IsCompact (spectrum NNReal a)
  -/
  rw [← preimage_algebraMap ℝ]
  /-
    𝕜 : Type u_1
    A✝ : Type u_2
    inst✝⁶ : NormedField 𝕜
    inst✝⁵ : NormedRing A✝
    inst✝⁴ : NormedAlgebra 𝕜 A✝
    inst✝³ : CompleteSpace A✝
    A : Type u_3
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    a : A
    inst✝ : IsCompact (spectrum Real a)
    ⊢ IsCompact (Set.preimage (⇑(algebraMap NNReal Real)) (spectrum Real a))
  -/
  exact isClosed_nonneg.isClosedEmbedding_subtypeVal.isCompact_preimage <| by assumption
  /-
    🎉 no goals
  -/


theorem _root_.quasispectrum.isCompact (a : B) : IsCompact (quasispectrum 𝕜 a) := by
  rw [Unitization.quasispectrum_eq_spectrum_inr' 𝕜 𝕜,
    ← AlgEquiv.spectrum_eq (WithLp.unitizationAlgEquiv 𝕜).symm (a : Unitization 𝕜 B)]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NormedField 𝕜
    B : Type u_3
    inst✝⁵ : NonUnitalNormedRing B
    inst✝⁴ : NormedSpace 𝕜 B
    inst✝³ : CompleteSpace B
    inst✝² : IsScalarTower 𝕜 B B
    inst✝¹ : SMulCommClass 𝕜 B B
    inst✝ : ProperSpace 𝕜
    a : B
    ⊢ IsCompact (spectrum 𝕜 ((WithLp.unitizationAlgEquiv 𝕜).symm ↑a))
  -/
  exact spectrum.isCompact _
  /-
    🎉 no goals
  -/


instance _root_.quasispectrum.instCompactSpace (a : B) :
    CompactSpace (quasispectrum 𝕜 a) :=
  isCompact_iff_compactSpace.mp <| quasispectrum.isCompact a


instance _root_.quasispectrum.instCompactSpaceNNReal [NormedSpace ℝ B] [IsScalarTower ℝ B B]
    [SMulCommClass ℝ B B] (a : B) [CompactSpace (quasispectrum ℝ a)] :
    CompactSpace (quasispectrum ℝ≥0 a) := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝¹³ : NormedField 𝕜
    inst✝¹² : NormedRing A
    inst✝¹¹ : NormedAlgebra 𝕜 A
    inst✝¹⁰ : CompleteSpace A
    B : Type u_3
    inst✝⁹ : NonUnitalNormedRing B
    inst✝⁸ : NormedSpace 𝕜 B
    inst✝⁷ : CompleteSpace B
    inst✝⁶ : IsScalarTower 𝕜 B B
    inst✝⁵ : SMulCommClass 𝕜 B B
    inst✝⁴ : ProperSpace 𝕜
    inst✝³ : NormedSpace Real B
    inst✝² : IsScalarTower Real B B
    inst✝¹ : SMulCommClass Real B B
    a : B
    inst✝ : CompactSpace ↑(quasispectrum Real a)
    ⊢ CompactSpace ↑(quasispectrum NNReal a)
  -/
  rw [← isCompact_iff_compactSpace] at *
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝¹³ : NormedField 𝕜
    inst✝¹² : NormedRing A
    inst✝¹¹ : NormedAlgebra 𝕜 A
    inst✝¹⁰ : CompleteSpace A
    B : Type u_3
    inst✝⁹ : NonUnitalNormedRing B
    inst✝⁸ : NormedSpace 𝕜 B
    inst✝⁷ : CompleteSpace B
    inst✝⁶ : IsScalarTower 𝕜 B B
    inst✝⁵ : SMulCommClass 𝕜 B B
    inst✝⁴ : ProperSpace 𝕜
    inst✝³ : NormedSpace Real B
    inst✝² : IsScalarTower Real B B
    inst✝¹ : SMulCommClass Real B B
    a : B
    inst✝ : IsCompact (quasispectrum Real a)
    ⊢ IsCompact (quasispectrum NNReal a)
  -/
  rw [← quasispectrum.preimage_algebraMap ℝ]
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝¹³ : NormedField 𝕜
    inst✝¹² : NormedRing A
    inst✝¹¹ : NormedAlgebra 𝕜 A
    inst✝¹⁰ : CompleteSpace A
    B : Type u_3
    inst✝⁹ : NonUnitalNormedRing B
    inst✝⁸ : NormedSpace 𝕜 B
    inst✝⁷ : CompleteSpace B
    inst✝⁶ : IsScalarTower 𝕜 B B
    inst✝⁵ : SMulCommClass 𝕜 B B
    inst✝⁴ : ProperSpace 𝕜
    inst✝³ : NormedSpace Real B
    inst✝² : IsScalarTower Real B B
    inst✝¹ : SMulCommClass Real B B
    a : B
    inst✝ : IsCompact (quasispectrum Real a)
    ⊢ IsCompact (Set.preimage (⇑(algebraMap NNReal Real)) (quasispectrum Real a))
  -/
  exact isClosed_nonneg.isClosedEmbedding_subtypeVal.isCompact_preimage <| by assumption
  /-
    🎉 no goals
  -/


theorem le_nnnorm_of_mem {a : A} {r : ℝ≥0} (hr : r ∈ spectrum ℝ≥0 a) :
    r ≤ ‖a‖₊ := calc
  r ≤ ‖(r : ℝ)‖ := Real.le_norm_self _
  _ ≤ ‖a‖       := norm_le_norm_of_mem hr


theorem coe_le_norm_of_mem {a : A} {r : ℝ≥0} (hr : r ∈ spectrum ℝ≥0 a) :
    r ≤ ‖a‖ :=
  coe_mono <| le_nnnorm_of_mem hr


theorem spectralRadius_le_nnnorm [NormOneClass A] (a : A) : spectralRadius 𝕜 a ≤ ‖a‖₊ := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁴ : NormedField 𝕜
    inst✝³ : NormedRing A
    inst✝² : NormedAlgebra 𝕜 A
    inst✝¹ : CompleteSpace A
    inst✝ : NormOneClass A
    a : A
    ⊢ LE.le (spectralRadius 𝕜 a) ↑(NNNorm.nnnorm a)
  -/
  refine iSup₂_le fun k hk => ?_
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁴ : NormedField 𝕜
    inst✝³ : NormedRing A
    inst✝² : NormedAlgebra 𝕜 A
    inst✝¹ : CompleteSpace A
    inst✝ : NormOneClass A
    a : A
    k : 𝕜
    hk : Membership.mem (spectrum 𝕜 a) k
    ⊢ LE.le ↑(NNNorm.nnnorm k) ↑(NNNorm.nnnorm a)
  -/
  exact mod_cast norm_le_norm_of_mem hk
  /-
    🎉 no goals
  -/


theorem exists_nnnorm_eq_spectralRadius_of_nonempty [ProperSpace 𝕜] {a : A} (ha : (σ a).Nonempty) :
    ∃ k ∈ σ a, (‖k‖₊ : ℝ≥0∞) = spectralRadius 𝕜 a := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁴ : NormedField 𝕜
    inst✝³ : NormedRing A
    inst✝² : NormedAlgebra 𝕜 A
    inst✝¹ : CompleteSpace A
    inst✝ : ProperSpace 𝕜
    a : A
    ha : (spectrum 𝕜 a).Nonempty
    ⊢ Exists fun k => And (Membership.mem (spectrum 𝕜 a) k) (Eq (↑(NNNorm.nnnorm k …
  -/
  obtain ⟨k, hk, h⟩ := (spectrum.isCompact a).exists_isMaxOn ha continuous_nnnorm.continuousOn
  /-
    case intro.intro
    𝕜 : Type u_1
    A : Type u_2
    inst✝⁴ : NormedField 𝕜
    inst✝³ : NormedRing A
    inst✝² : NormedAlgebra 𝕜 A
    inst✝¹ : CompleteSpace A
    inst✝ : ProperSpace 𝕜
    a : A
    ha : (spectrum 𝕜 a).Nonempty
    k : 𝕜
    hk : Membership.mem (spectrum 𝕜 a) k
    h : IsMaxOn (fun a => NNNorm.nnnorm a) (spectrum 𝕜 a) k
    ⊢ Exists fun k => And (Membership.mem (spectrum 𝕜 a) k) (Eq (↑(NNNorm.nnnorm k …
  -/
  exact ⟨k, hk, le_antisymm (le_iSup₂ (α := ℝ≥0∞) k hk) (iSup₂_le <| mod_cast h)⟩
  /-
    🎉 no goals
  -/


theorem spectralRadius_lt_of_forall_lt_of_nonempty [ProperSpace 𝕜] {a : A} (ha : (σ a).Nonempty)
    {r : ℝ≥0} (hr : ∀ k ∈ σ a, ‖k‖₊ < r) : spectralRadius 𝕜 a < r :=
  sSup_image.symm.trans_lt <|
    ((spectrum.isCompact a).sSup_lt_iff_of_continuous ha
          (ENNReal.continuous_coe.comp continuous_nnnorm).continuousOn (r : ℝ≥0∞)).mpr
          /-
            𝕜 : Type u_1
            A : Type u_2
            inst✝⁴ : NormedField 𝕜
            inst✝³ : NormedRing A
            inst✝² : NormedAlgebra 𝕜 A
            inst✝¹ : CompleteSpace A
            inst✝ : ProperSpace 𝕜
            a : A
            ha : (spectrum 𝕜 a).Nonempty
            r : NNReal
            hr : ∀ (k : 𝕜), Membership.mem (spectrum 𝕜 a) k → LT.lt (NNNorm.nnnorm k) r
            ⊢ ∀ (x : 𝕜), Membership.mem (spectrum 𝕜 a) x → LT.lt (Function.comp ENNReal.of …
          -/
      (by dsimp only [(· ∘ ·)]; exact mod_cast hr)
                                /-
                                  🎉 no goals
                                -/


theorem spectralRadius_le_pow_nnnorm_pow_one_div (a : A) (n : ℕ) :
    spectralRadius 𝕜 a ≤ (‖a ^ (n + 1)‖₊ : ℝ≥0∞) ^ (1 / (n + 1) : ℝ) *
      (‖(1 : A)‖₊ : ℝ≥0∞) ^ (1 / (n + 1) : ℝ) := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    n : Nat
    ⊢ LE.le (spectralRadius 𝕜 a) (HMul.hMul (HPow.hPow (↑(NNNorm.nnnorm (HPow.hPow …
  -/
  refine iSup₂_le fun k hk => ?_
  -- apply easy direction of the spectral mapping theorem for polynomials
  have pow_mem : k ^ (n + 1) ∈ σ (a ^ (n + 1)) := by
    simpa only [one_mul, Algebra.algebraMap_eq_smul_one, one_smul, aeval_monomial, one_mul,
      eval_monomial] using subset_polynomial_aeval a (@monomial 𝕜 _ (n + 1) (1 : 𝕜)) ⟨k, hk, rfl⟩
  -- power of the norm is bounded by norm of the power
  have nnnorm_pow_le : (↑(‖k‖₊ ^ (n + 1)) : ℝ≥0∞) ≤ ‖a ^ (n + 1)‖₊ * ‖(1 : A)‖₊ := by
    simpa only [Real.toNNReal_mul (norm_nonneg _), norm_toNNReal, nnnorm_pow k (n + 1),
      ENNReal.coe_mul] using coe_mono (Real.toNNReal_mono (norm_le_norm_mul_of_mem pow_mem))
  -- take (n + 1)ᵗʰ roots and clean up the left-hand side
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    n : Nat
    k : 𝕜
    hk : Membership.mem (spectrum 𝕜 a) k
    pow_mem : Membership.mem (spectrum 𝕜 (HPow.hPow a (HAdd.hAdd n 1))) (HPow.hPow …
    nnnorm_pow_le : LE.le (↑(HPow.hPow (NNNorm.nnnorm k) (HAdd.hAdd n 1))) (HMul.h …
    ⊢ LE.le (↑(NNNorm.nnnorm k)) (HMul.hMul (HPow.hPow (↑(NNNorm.nnnorm (HPow.hPow …
  -/
  have hn : 0 < ((n + 1 : ℕ) : ℝ) := mod_cast Nat.succ_pos'
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    n : Nat
    k : 𝕜
    hk : Membership.mem (spectrum 𝕜 a) k
    pow_mem : Membership.mem (spectrum 𝕜 (HPow.hPow a (HAdd.hAdd n 1))) (HPow.hPow …
    nnnorm_pow_le : LE.le (↑(HPow.hPow (NNNorm.nnnorm k) (HAdd.hAdd n 1))) (HMul.h …
    hn : LT.lt 0 ↑(HAdd.hAdd n 1)
    ⊢ LE.le (↑(NNNorm.nnnorm k)) (HMul.hMul (HPow.hPow (↑(NNNorm.nnnorm (HPow.hPow …
  -/
  convert monotone_rpow_of_nonneg (one_div_pos.mpr hn).le nnnorm_pow_le using 1
  /-
    case h.e'_3
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    n : Nat
    k : 𝕜
    hk : Membership.mem (spectrum 𝕜 a) k
    pow_mem : Membership.mem (spectrum 𝕜 (HPow.hPow a (HAdd.hAdd n 1))) (HPow.hPow …
    nnnorm_pow_le : LE.le (↑(HPow.hPow (NNNorm.nnnorm k) (HAdd.hAdd n 1))) (HMul.h …
    hn : LT.lt 0 ↑(HAdd.hAdd n 1)
    ⊢ Eq (↑(NNNorm.nnnorm k)) ((fun x => HPow.hPow x (HDiv.hDiv 1 ↑(HAdd.hAdd n 1) …
  -/
  all_goals dsimp
    /-
      case h.e'_3
      𝕜 : Type u_1
      A : Type u_2
      inst✝³ : NormedField 𝕜
      inst✝² : NormedRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : CompleteSpace A
      a : A
      n : Nat
      k : 𝕜
      hk : Membership.mem (spectrum 𝕜 a) k
      pow_mem : Membership.mem (spectrum 𝕜 (HPow.hPow a (HAdd.hAdd n 1))) (HPow.hPow …
      nnnorm_pow_le : LE.le (↑(HPow.hPow (NNNorm.nnnorm k) (HAdd.hAdd n 1))) (HMul.h …
      hn : LT.lt 0 ↑(HAdd.hAdd n 1)
      ⊢ Eq (↑(NNNorm.nnnorm k)) (HPow.hPow (HPow.hPow (↑(NNNorm.nnnorm k)) (HAdd.hAd …
    -/
  · rw [one_div, pow_rpow_inv_natCast]
    /-
      case h.e'_3.hn
      𝕜 : Type u_1
      A : Type u_2
      inst✝³ : NormedField 𝕜
      inst✝² : NormedRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : CompleteSpace A
      a : A
      n : Nat
      k : 𝕜
      hk : Membership.mem (spectrum 𝕜 a) k
      pow_mem : Membership.mem (spectrum 𝕜 (HPow.hPow a (HAdd.hAdd n 1))) (HPow.hPow …
      nnnorm_pow_le : LE.le (↑(HPow.hPow (NNNorm.nnnorm k) (HAdd.hAdd n 1))) (HMul.h …
      hn : LT.lt 0 ↑(HAdd.hAdd n 1)
      ⊢ Ne (HAdd.hAdd n 1) 0
    -/
    positivity
    /-
      🎉 no goals
    -/
  /-
    case h.e'_4
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    n : Nat
    k : 𝕜
    hk : Membership.mem (spectrum 𝕜 a) k
    pow_mem : Membership.mem (spectrum 𝕜 (HPow.hPow a (HAdd.hAdd n 1))) (HPow.hPow …
    nnnorm_pow_le : LE.le (↑(HPow.hPow (NNNorm.nnnorm k) (HAdd.hAdd n 1))) (HMul.h …
    hn : LT.lt 0 ↑(HAdd.hAdd n 1)
    ⊢ Eq (HMul.hMul (HPow.hPow (↑(NNNorm.nnnorm (HPow.hPow a (HAdd.hAdd n 1)))) (H …
  -/
  rw [Nat.cast_succ, ENNReal.coe_mul_rpow]
  /-
    🎉 no goals
  -/


theorem spectralRadius_le_liminf_pow_nnnorm_pow_one_div (a : A) :
    spectralRadius 𝕜 a ≤ atTop.liminf fun n : ℕ => (‖a ^ n‖₊ : ℝ≥0∞) ^ (1 / n : ℝ) := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    ⊢ LE.le (spectralRadius 𝕜 a) (Filter.liminf (fun n => HPow.hPow (↑(NNNorm.nnno …
  -/
  refine ENNReal.le_of_forall_lt_one_mul_le fun ε hε => ?_
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    ε : ENNReal
    hε : LT.lt ε 1
    ⊢ LE.le (HMul.hMul ε (spectralRadius 𝕜 a)) (Filter.liminf (fun n => HPow.hPow  …
  -/
  by_cases h : ε = 0
    /-
      case pos
      𝕜 : Type u_1
      A : Type u_2
      inst✝³ : NormedField 𝕜
      inst✝² : NormedRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : CompleteSpace A
      a : A
      ε : ENNReal
      hε : LT.lt ε 1
      h : Eq ε 0
      ⊢ LE.le (HMul.hMul ε (spectralRadius 𝕜 a)) (Filter.liminf (fun n => HPow.hPow  …
    -/
  · simp only [h, zero_mul, zero_le']
    /-
      🎉 no goals
    -/
  simp only [ENNReal.mul_le_iff_le_inv h (hε.trans_le le_top).ne, mul_comm ε⁻¹,
    liminf_eq_iSup_iInf_of_nat', ENNReal.iSup_mul]
  /-
    case neg
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    ε : ENNReal
    hε : LT.lt ε 1
    h : Not (Eq ε 0)
    ⊢ LE.le (spectralRadius 𝕜 a) (iSup fun i => HMul.hMul (iInf fun i_1 => HPow.hP …
  -/
  conv_rhs => arg 1; intro i; rw [ENNReal.iInf_mul (by simp [h])]
  /-
    case neg
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    ε : ENNReal
    hε : LT.lt ε 1
    h : Not (Eq ε 0)
    ⊢ LE.le (spectralRadius 𝕜 a) (iSup fun i => iInf fun i_1 => HMul.hMul (HPow.hP …
  -/
  rw [← ENNReal.inv_lt_inv, inv_one] at hε
  obtain ⟨N, hN⟩ := eventually_atTop.mp
    (ENNReal.eventually_pow_one_div_le (ENNReal.coe_ne_top : ↑‖(1 : A)‖₊ ≠ ∞) hε)
  /-
    case neg.intro
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    ε : ENNReal
    hε : LT.lt 1 (Inv.inv ε)
    h : Not (Eq ε 0)
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → LE.le (HPow.hPow (↑(NNNorm.nnnorm 1)) (HDiv.hDiv …
    ⊢ LE.le (spectralRadius 𝕜 a) (iSup fun i => iInf fun i_1 => HMul.hMul (HPow.hP …
  -/
  refine le_trans ?_ (le_iSup _ (N + 1))
  /-
    case neg.intro
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    ε : ENNReal
    hε : LT.lt 1 (Inv.inv ε)
    h : Not (Eq ε 0)
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → LE.le (HPow.hPow (↑(NNNorm.nnnorm 1)) (HDiv.hDiv …
    ⊢ LE.le (spectralRadius 𝕜 a) (iInf fun i => HMul.hMul (HPow.hPow (↑(NNNorm.nnn …
  -/
  refine le_iInf fun n => ?_
  /-
    case neg.intro
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    ε : ENNReal
    hε : LT.lt 1 (Inv.inv ε)
    h : Not (Eq ε 0)
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → LE.le (HPow.hPow (↑(NNNorm.nnnorm 1)) (HDiv.hDiv …
    n : Nat
    ⊢ LE.le (spectralRadius 𝕜 a) (HMul.hMul (HPow.hPow (↑(NNNorm.nnnorm (HPow.hPow …
  -/
  simp only [← add_assoc]
  /-
    case neg.intro
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    ε : ENNReal
    hε : LT.lt 1 (Inv.inv ε)
    h : Not (Eq ε 0)
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → LE.le (HPow.hPow (↑(NNNorm.nnnorm 1)) (HDiv.hDiv …
    n : Nat
    ⊢ LE.le (spectralRadius 𝕜 a) (HMul.hMul (HPow.hPow (↑(NNNorm.nnnorm (HPow.hPow …
  -/
  refine (spectralRadius_le_pow_nnnorm_pow_one_div 𝕜 a (n + N)).trans ?_
  /-
    case neg.intro
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    ε : ENNReal
    hε : LT.lt 1 (Inv.inv ε)
    h : Not (Eq ε 0)
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → LE.le (HPow.hPow (↑(NNNorm.nnnorm 1)) (HDiv.hDiv …
    n : Nat
    ⊢ LE.le (HMul.hMul (HPow.hPow (↑(NNNorm.nnnorm (HPow.hPow a (HAdd.hAdd (HAdd.h …
  -/
  norm_cast
  /-
    case neg.intro
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    ε : ENNReal
    hε : LT.lt 1 (Inv.inv ε)
    h : Not (Eq ε 0)
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → LE.le (HPow.hPow (↑(NNNorm.nnnorm 1)) (HDiv.hDiv …
    n : Nat
    ⊢ LE.le (HMul.hMul (HPow.hPow (↑(NNNorm.nnnorm (HPow.hPow a (HAdd.hAdd (HAdd.h …
  -/
  exact mul_le_mul_left' (hN (n + N + 1) (by omega)) _
  /-
    🎉 no goals
  -/


theorem hasDerivAt_resolvent {a : A} {k : 𝕜} (hk : k ∈ ρ a) :
    HasDerivAt (resolvent a) (-resolvent a k ^ 2) k := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    k : 𝕜
    hk : Membership.mem (resolventSet 𝕜 a) k
    ⊢ HasDerivAt (resolvent a) (Neg.neg (HPow.hPow (resolvent a k) 2)) k
  -/
  have H₁ : HasFDerivAt Ring.inverse _ (↑ₐ k - a) := hasFDerivAt_ring_inverse (𝕜 := 𝕜) hk.unit
  have H₂ : HasDerivAt (fun k => ↑ₐ k - a) 1 k := by
    simpa using (Algebra.linearMap 𝕜 A).hasDerivAt.sub_const a
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    k : 𝕜
    hk : Membership.mem (resolventSet 𝕜 a) k
    H₁ : HasFDerivAt Ring.inverse (Neg.neg (((ContinuousLinearMap.mulLeftRight 𝕜 A …
    H₂ : HasDerivAt (fun k => HSub.hSub ((algebraMap 𝕜 A) k) a) 1 k
    ⊢ HasDerivAt (resolvent a) (Neg.neg (HPow.hPow (resolvent a k) 2)) k
  -/
  simpa [resolvent, sq, hk.unit_spec, ← Ring.inverse_unit hk.unit] using H₁.comp_hasDerivAt k H₂
  /-
    🎉 no goals
  -/

-- refactored so this result was no longer necessary or useful


theorem eventually_isUnit_resolvent (a : A) : ∀ᶠ z in cobounded 𝕜, IsUnit (resolvent a z) := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    ⊢ Filter.Eventually (fun z => IsUnit (resolvent a z)) (Bornology.cobounded 𝕜)
  -/
  rw [atTop_basis_Ioi.cobounded_of_norm.eventually_iff]
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    ⊢ Exists fun i => And True (∀ ⦃x : 𝕜⦄, Membership.mem (Set.preimage Norm.norm  …
  -/
  exact ⟨‖a‖ * ‖(1 : A)‖, trivial, fun _ ↦ isUnit_resolvent.mp ∘ mem_resolventSet_of_norm_lt_mul⟩
  /-
    🎉 no goals
  -/


theorem resolvent_isBigO_inv (a : A) : resolvent a =O[cobounded 𝕜] Inv.inv :=
  have h : (fun z ↦ resolvent (z⁻¹ • a) (1 : 𝕜)) =O[cobounded 𝕜] (fun _ ↦ (1 : ℝ)) := by
    simpa [Function.comp_def, resolvent] using
      (NormedRing.inverse_one_sub_norm (R := A)).comp_tendsto
        (by simpa using (tendsto_inv₀_cobounded (α := 𝕜)).smul_const a)
  calc
    resolvent a =ᶠ[cobounded 𝕜] fun z ↦ z⁻¹ • resolvent (z⁻¹ • a) (1 : 𝕜) := by
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing A
        inst✝¹ : NormedAlgebra 𝕜 A
        inst✝ : CompleteSpace A
        a : A
        h : Asymptotics.IsBigO (Bornology.cobounded 𝕜) (fun z => resolvent (HSMul.hSMu …
        ⊢ (Bornology.cobounded 𝕜).EventuallyEq (resolvent a) fun z => HSMul.hSMul (Inv …
      -/
      filter_upwards [isBounded_singleton (x := 0)] with z hz
      /-
        case h
        𝕜 : Type u_1
        A : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing A
        inst✝¹ : NormedAlgebra 𝕜 A
        inst✝ : CompleteSpace A
        a : A
        h : Asymptotics.IsBigO (Bornology.cobounded 𝕜) (fun z => resolvent (HSMul.hSMu …
        z : 𝕜
        hz : Membership.mem (HasCompl.compl (Singleton.singleton 0)) z
        ⊢ Eq (resolvent a z) (HSMul.hSMul (Inv.inv z) (resolvent (HSMul.hSMul (Inv.inv …
      -/
      lift z to 𝕜ˣ using Ne.isUnit hz
      /-
        case h.intro
        𝕜 : Type u_1
        A : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing A
        inst✝¹ : NormedAlgebra 𝕜 A
        inst✝ : CompleteSpace A
        a : A
        h : Asymptotics.IsBigO (Bornology.cobounded 𝕜) (fun z => resolvent (HSMul.hSMu …
        z : Units 𝕜
        hz : Membership.mem (HasCompl.compl (Singleton.singleton 0)) ↑z
        ⊢ Eq (resolvent a ↑z) (HSMul.hSMul (Inv.inv ↑z) (resolvent (HSMul.hSMul (Inv.i …
      -/
      simpa [Units.smul_def] using congr(z⁻¹ • $(units_smul_resolvent_self (r := z) (a := a)))
      /-
        🎉 no goals
      -/
    _ =O[cobounded 𝕜] (· ⁻¹) := .of_norm_right <| by
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing A
        inst✝¹ : NormedAlgebra 𝕜 A
        inst✝ : CompleteSpace A
        a : A
        h : Asymptotics.IsBigO (Bornology.cobounded 𝕜) (fun z => resolvent (HSMul.hSMu …
        ⊢ Asymptotics.IsBigO (Bornology.cobounded 𝕜) (fun z => HSMul.hSMul (Inv.inv z) …
      -/
      simpa using (isBigO_refl (· ⁻¹) (cobounded 𝕜)).norm_right.smul h
      /-
        🎉 no goals
      -/


theorem resolvent_tendsto_cobounded (a : A) : Tendsto (resolvent a) (cobounded 𝕜) (𝓝 0) :=
  resolvent_isBigO_inv a |>.trans_tendsto tendsto_inv₀_cobounded


/-- In a Banach algebra `A` over a nontrivially normed field `𝕜`, for any `a : A` the
power series with coefficients `a ^ n` represents the function `(1 - z • a)⁻¹` in a disk of
radius `‖a‖₊⁻¹`. -/
theorem hasFPowerSeriesOnBall_inverse_one_sub_smul [HasSummableGeomSeries A] (a : A) :
    HasFPowerSeriesOnBall (fun z : 𝕜 => Ring.inverse (1 - z • a))
      (fun n => ContinuousMultilinearMap.mkPiRing 𝕜 (Fin n) (a ^ n)) 0 ‖a‖₊⁻¹ :=
  { r_le := by
      refine le_of_forall_nnreal_lt fun r hr =>
        le_radius_of_bound_nnreal _ (max 1 ‖(1 : A)‖₊) fun n => ?_
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing A
        inst✝¹ : NormedAlgebra 𝕜 A
        inst✝ : HasSummableGeomSeries A
        a : A
        r : NNReal
        hr : LT.lt (↑r) (Inv.inv ↑(NNNorm.nnnorm a))
        n : Nat
        ⊢ LE.le (HMul.hMul (NNNorm.nnnorm (ContinuousMultilinearMap.mkPiRing 𝕜 (Fin n) …
      -/
      rw [← norm_toNNReal, norm_mkPiRing, norm_toNNReal]
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing A
        inst✝¹ : NormedAlgebra 𝕜 A
        inst✝ : HasSummableGeomSeries A
        a : A
        r : NNReal
        hr : LT.lt (↑r) (Inv.inv ↑(NNNorm.nnnorm a))
        n : Nat
        ⊢ LE.le (HMul.hMul (NNNorm.nnnorm (HPow.hPow a n)) (HPow.hPow r n)) (Max.max 1 …
      -/
      cases' n with n
        /-
          case zero
          𝕜 : Type u_1
          A : Type u_2
          inst✝³ : NontriviallyNormedField 𝕜
          inst✝² : NormedRing A
          inst✝¹ : NormedAlgebra 𝕜 A
          inst✝ : HasSummableGeomSeries A
          a : A
          r : NNReal
          hr : LT.lt (↑r) (Inv.inv ↑(NNNorm.nnnorm a))
          ⊢ LE.le (HMul.hMul (NNNorm.nnnorm (HPow.hPow a 0)) (HPow.hPow r 0)) (Max.max 1 …
        -/
      · simp only [le_refl, mul_one, or_true, le_max_iff, pow_zero]
        /-
          🎉 no goals
        -/
      · refine
          le_trans (le_trans (mul_le_mul_right' (nnnorm_pow_le' a n.succ_pos) (r ^ n.succ)) ?_)
            (le_max_left _ _)
        /-
          case succ
          𝕜 : Type u_1
          A : Type u_2
          inst✝³ : NontriviallyNormedField 𝕜
          inst✝² : NormedRing A
          inst✝¹ : NormedAlgebra 𝕜 A
          inst✝ : HasSummableGeomSeries A
          a : A
          r : NNReal
          hr : LT.lt (↑r) (Inv.inv ↑(NNNorm.nnnorm a))
          n : Nat
          ⊢ LE.le (HMul.hMul (HPow.hPow (NNNorm.nnnorm a) n.succ) (HPow.hPow r n.succ)) 1
        -/
        by_cases h : ‖a‖₊ = 0
          /-
            case pos
            𝕜 : Type u_1
            A : Type u_2
            inst✝³ : NontriviallyNormedField 𝕜
            inst✝² : NormedRing A
            inst✝¹ : NormedAlgebra 𝕜 A
            inst✝ : HasSummableGeomSeries A
            a : A
            r : NNReal
            hr : LT.lt (↑r) (Inv.inv ↑(NNNorm.nnnorm a))
            n : Nat
            h : Eq (NNNorm.nnnorm a) 0
            ⊢ LE.le (HMul.hMul (HPow.hPow (NNNorm.nnnorm a) n.succ) (HPow.hPow r n.succ)) 1
          -/
        · simp only [h, zero_mul, zero_le', pow_succ']
          /-
            🎉 no goals
          -/
          /-
            case neg
            𝕜 : Type u_1
            A : Type u_2
            inst✝³ : NontriviallyNormedField 𝕜
            inst✝² : NormedRing A
            inst✝¹ : NormedAlgebra 𝕜 A
            inst✝ : HasSummableGeomSeries A
            a : A
            r : NNReal
            hr : LT.lt (↑r) (Inv.inv ↑(NNNorm.nnnorm a))
            n : Nat
            h : Not (Eq (NNNorm.nnnorm a) 0)
            ⊢ LE.le (HMul.hMul (HPow.hPow (NNNorm.nnnorm a) n.succ) (HPow.hPow r n.succ)) 1
          -/
        · rw [← coe_inv h, coe_lt_coe, NNReal.lt_inv_iff_mul_lt h] at hr
          /-
            case neg
            𝕜 : Type u_1
            A : Type u_2
            inst✝³ : NontriviallyNormedField 𝕜
            inst✝² : NormedRing A
            inst✝¹ : NormedAlgebra 𝕜 A
            inst✝ : HasSummableGeomSeries A
            a : A
            r : NNReal
            hr : LT.lt (HMul.hMul r (NNNorm.nnnorm a)) 1
            n : Nat
            h : Not (Eq (NNNorm.nnnorm a) 0)
            ⊢ LE.le (HMul.hMul (HPow.hPow (NNNorm.nnnorm a) n.succ) (HPow.hPow r n.succ)) 1
          -/
          simpa only [← mul_pow, mul_comm] using pow_le_one' hr.le n.succ
          /-
            🎉 no goals
          -/
    r_pos := ENNReal.inv_pos.mpr coe_ne_top
    hasSum := fun {y} hy => by
      have norm_lt : ‖y • a‖ < 1 := by
        by_cases h : ‖a‖₊ = 0
        · simp only [nnnorm_eq_zero.mp h, norm_zero, zero_lt_one, smul_zero]
        · have nnnorm_lt : ‖y‖₊ < ‖a‖₊⁻¹ := by
            simpa only [← coe_inv h, mem_ball_zero_iff, Metric.emetric_ball_nnreal] using hy
          rwa [← coe_nnnorm, ← Real.lt_toNNReal_iff_coe_lt, Real.toNNReal_one, nnnorm_smul,
            ← NNReal.lt_inv_iff_mul_lt h]
      simpa [← smul_pow, (summable_geometric_of_norm_lt_one norm_lt).hasSum_iff] using
        (NormedRing.inverse_one_sub _ norm_lt).symm }


theorem isUnit_one_sub_smul_of_lt_inv_radius {a : A} {z : 𝕜} (h : ↑‖z‖₊ < (spectralRadius 𝕜 a)⁻¹) :
    IsUnit (1 - z • a) := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedRing A
    inst✝ : NormedAlgebra 𝕜 A
    a : A
    z : 𝕜
    h : LT.lt (↑(NNNorm.nnnorm z)) (Inv.inv (spectralRadius 𝕜 a))
    ⊢ IsUnit (HSub.hSub 1 (HSMul.hSMul z a))
  -/
  by_cases hz : z = 0
    /-
      case pos
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedRing A
      inst✝ : NormedAlgebra 𝕜 A
      a : A
      z : 𝕜
      h : LT.lt (↑(NNNorm.nnnorm z)) (Inv.inv (spectralRadius 𝕜 a))
      hz : Eq z 0
      ⊢ IsUnit (HSub.hSub 1 (HSMul.hSMul z a))
    -/
  · simp only [hz, isUnit_one, sub_zero, zero_smul]
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedRing A
      inst✝ : NormedAlgebra 𝕜 A
      a : A
      z : 𝕜
      h : LT.lt (↑(NNNorm.nnnorm z)) (Inv.inv (spectralRadius 𝕜 a))
      hz : Not (Eq z 0)
      ⊢ IsUnit (HSub.hSub 1 (HSMul.hSMul z a))
    -/
  · let u := Units.mk0 z hz
    suffices hu : IsUnit (u⁻¹ • (1 : A) - a) by
      rwa [IsUnit.smul_sub_iff_sub_inv_smul, inv_inv u] at hu
    /-
      case neg
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedRing A
      inst✝ : NormedAlgebra 𝕜 A
      a : A
      z : 𝕜
      h : LT.lt (↑(NNNorm.nnnorm z)) (Inv.inv (spectralRadius 𝕜 a))
      hz : Not (Eq z 0)
      u : Units 𝕜 := Units.mk0 z hz
      ⊢ IsUnit (HSub.hSub (HSMul.hSMul (Inv.inv u) 1) a)
    -/
    rw [Units.smul_def, ← Algebra.algebraMap_eq_smul_one, ← mem_resolventSet_iff]
    /-
      case neg
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedRing A
      inst✝ : NormedAlgebra 𝕜 A
      a : A
      z : 𝕜
      h : LT.lt (↑(NNNorm.nnnorm z)) (Inv.inv (spectralRadius 𝕜 a))
      hz : Not (Eq z 0)
      u : Units 𝕜 := Units.mk0 z hz
      ⊢ Membership.mem (resolventSet 𝕜 a) ↑(Inv.inv u)
    -/
    refine mem_resolventSet_of_spectralRadius_lt ?_
    rwa [Units.val_inv_eq_inv_val, nnnorm_inv,
      coe_inv (nnnorm_ne_zero_iff.mpr (Units.val_mk0 hz ▸ hz : (u : 𝕜) ≠ 0)), lt_inv_iff_lt_inv]


/-- In a Banach algebra `A` over `𝕜`, for `a : A` the function `fun z ↦ (1 - z • a)⁻¹` is
differentiable on any closed ball centered at zero of radius `r < (spectralRadius 𝕜 a)⁻¹`. -/
theorem differentiableOn_inverse_one_sub_smul [CompleteSpace A] {a : A} {r : ℝ≥0}
    (hr : (r : ℝ≥0∞) < (spectralRadius 𝕜 a)⁻¹) :
    DifferentiableOn 𝕜 (fun z : 𝕜 => Ring.inverse (1 - z • a)) (Metric.closedBall 0 r) := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    r : NNReal
    hr : LT.lt (↑r) (Inv.inv (spectralRadius 𝕜 a))
    ⊢ DifferentiableOn 𝕜 (fun z => Ring.inverse (HSub.hSub 1 (HSMul.hSMul z a))) ( …
  -/
  intro z z_mem
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    r : NNReal
    hr : LT.lt (↑r) (Inv.inv (spectralRadius 𝕜 a))
    z : 𝕜
    z_mem : Membership.mem (Metric.closedBall 0 ↑r) z
    ⊢ DifferentiableWithinAt 𝕜 (fun z => Ring.inverse (HSub.hSub 1 (HSMul.hSMul z  …
  -/
  apply DifferentiableAt.differentiableWithinAt
  have hu : IsUnit (1 - z • a) := by
    refine isUnit_one_sub_smul_of_lt_inv_radius (lt_of_le_of_lt (coe_mono ?_) hr)
    simpa only [norm_toNNReal, Real.toNNReal_coe] using
      Real.toNNReal_mono (mem_closedBall_zero_iff.mp z_mem)
  /-
    case h
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    r : NNReal
    hr : LT.lt (↑r) (Inv.inv (spectralRadius 𝕜 a))
    z : 𝕜
    z_mem : Membership.mem (Metric.closedBall 0 ↑r) z
    hu : IsUnit (HSub.hSub 1 (HSMul.hSMul z a))
    ⊢ DifferentiableAt 𝕜 (fun z => Ring.inverse (HSub.hSub 1 (HSMul.hSMul z a))) z
  -/
  have H₁ : Differentiable 𝕜 fun w : 𝕜 => 1 - w • a := (differentiable_id.smul_const a).const_sub 1
  /-
    case h
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    r : NNReal
    hr : LT.lt (↑r) (Inv.inv (spectralRadius 𝕜 a))
    z : 𝕜
    z_mem : Membership.mem (Metric.closedBall 0 ↑r) z
    hu : IsUnit (HSub.hSub 1 (HSMul.hSMul z a))
    H₁ : Differentiable 𝕜 fun w => HSub.hSub 1 (HSMul.hSMul w a)
    ⊢ DifferentiableAt 𝕜 (fun z => Ring.inverse (HSub.hSub 1 (HSMul.hSMul z a))) z
  -/
  exact DifferentiableAt.comp z (differentiableAt_inverse hu) H₁.differentiableAt
  /-
    🎉 no goals
  -/


/-- The `limsup` relationship for the spectral radius used to prove `spectrum.gelfand_formula`. -/
theorem limsup_pow_nnnorm_pow_one_div_le_spectralRadius (a : A) :
    limsup (fun n : ℕ => (‖a ^ n‖₊ : ℝ≥0∞) ^ (1 / n : ℝ)) atTop ≤ spectralRadius ℂ a := by
  /-
    A : Type u_2
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Complex A
    inst✝ : CompleteSpace A
    a : A
    ⊢ LE.le (Filter.limsup (fun n => HPow.hPow (↑(NNNorm.nnnorm (HPow.hPow a n)))  …
  -/
  refine ENNReal.inv_le_inv.mp (le_of_forall_pos_nnreal_lt fun r r_pos r_lt => ?_)
  /-
    A : Type u_2
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Complex A
    inst✝ : CompleteSpace A
    a : A
    r : NNReal
    r_pos : LT.lt 0 r
    r_lt : LT.lt (↑r) (Inv.inv (spectralRadius Complex a))
    ⊢ LE.le (↑r) (Inv.inv (Filter.limsup (fun n => HPow.hPow (↑(NNNorm.nnnorm (HPo …
  -/
  simp_rw [inv_limsup, ← one_div]
  let p : FormalMultilinearSeries ℂ ℂ A := fun n =>
    ContinuousMultilinearMap.mkPiRing ℂ (Fin n) (a ^ n)
  suffices h : (r : ℝ≥0∞) ≤ p.radius by
    convert h
    simp only [p, p.radius_eq_liminf, ← norm_toNNReal, norm_mkPiRing]
    congr
    ext n
    rw [norm_toNNReal, ENNReal.coe_rpow_def ‖a ^ n‖₊ (1 / n : ℝ), if_neg]
    exact fun ha => (lt_self_iff_false _).mp
      (ha.2.trans_le (one_div_nonneg.mpr n.cast_nonneg : 0 ≤ (1 / n : ℝ)))
  /-
    A : Type u_2
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Complex A
    inst✝ : CompleteSpace A
    a : A
    r : NNReal
    r_pos : LT.lt 0 r
    r_lt : LT.lt (↑r) (Inv.inv (spectralRadius Complex a))
    p : FormalMultilinearSeries Complex Complex A := fun n => ContinuousMultilinea …
    ⊢ LE.le (↑r) p.radius
  -/
  have H₁ := (differentiableOn_inverse_one_sub_smul r_lt).hasFPowerSeriesOnBall r_pos
  /-
    A : Type u_2
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Complex A
    inst✝ : CompleteSpace A
    a : A
    r : NNReal
    r_pos : LT.lt 0 r
    r_lt : LT.lt (↑r) (Inv.inv (spectralRadius Complex a))
    p : FormalMultilinearSeries Complex Complex A := fun n => ContinuousMultilinea …
    H₁ : HasFPowerSeriesOnBall (fun z => Ring.inverse (HSub.hSub 1 (HSMul.hSMul z  …
    ⊢ LE.le (↑r) p.radius
  -/
  exact ((hasFPowerSeriesOnBall_inverse_one_sub_smul ℂ a).exchange_radius H₁).r_le
  /-
    🎉 no goals
  -/


/-- **Gelfand's formula**: Given an element `a : A` of a complex Banach algebra, the
`spectralRadius` of `a` is the limit of the sequence `‖a ^ n‖₊ ^ (1 / n)`. -/
theorem pow_nnnorm_pow_one_div_tendsto_nhds_spectralRadius (a : A) :
    Tendsto (fun n : ℕ => (‖a ^ n‖₊ : ℝ≥0∞) ^ (1 / n : ℝ)) atTop (𝓝 (spectralRadius ℂ a)) :=
  /-
    A : Type u_2
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Complex A
    inst✝ : CompleteSpace A
    a : A
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun n => HPow. …
  -/
  /-
    🎉 no goals
  -/
  tendsto_of_le_liminf_of_limsup_le (spectralRadius_le_liminf_pow_nnnorm_pow_one_div ℂ a)
  /-
    🎉 no goals
  -/
    (limsup_pow_nnnorm_pow_one_div_le_spectralRadius a)

/- This is the same as `pow_nnnorm_pow_one_div_tendsto_nhds_spectralRadius` but for `norm`
instead of `nnnorm`. -/

/-- **Gelfand's formula**: Given an element `a : A` of a complex Banach algebra, the
`spectralRadius` of `a` is the limit of the sequence `‖a ^ n‖₊ ^ (1 / n)`. -/
theorem pow_norm_pow_one_div_tendsto_nhds_spectralRadius (a : A) :
    Tendsto (fun n : ℕ => ENNReal.ofReal (‖a ^ n‖ ^ (1 / n : ℝ))) atTop
      (𝓝 (spectralRadius ℂ a)) := by
  /-
    A : Type u_2
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Complex A
    inst✝ : CompleteSpace A
    a : A
    ⊢ Filter.Tendsto (fun n => ENNReal.ofReal (HPow.hPow (Norm.norm (HPow.hPow a n …
  -/
  convert pow_nnnorm_pow_one_div_tendsto_nhds_spectralRadius a using 1
  /-
    case h.e'_3
    A : Type u_2
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Complex A
    inst✝ : CompleteSpace A
    a : A
    ⊢ Eq (fun n => ENNReal.ofReal (HPow.hPow (Norm.norm (HPow.hPow a n)) (HDiv.hDi …
  -/
  ext1
  /-
    case h.e'_3.h
    A : Type u_2
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Complex A
    inst✝ : CompleteSpace A
    a : A
    x✝ : Nat
    ⊢ Eq (ENNReal.ofReal (HPow.hPow (Norm.norm (HPow.hPow a x✝)) (HDiv.hDiv 1 ↑x✝) …
  -/
  rw [← ofReal_rpow_of_nonneg (norm_nonneg _) _, ← coe_nnnorm, coe_nnreal_eq]
  /-
    A : Type u_2
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Complex A
    inst✝ : CompleteSpace A
    a : A
    x✝ : Nat
    ⊢ LE.le 0 (HDiv.hDiv 1 ↑x✝)
  -/
  exact one_div_nonneg.mpr (mod_cast zero_le _)
  /-
    🎉 no goals
  -/


/-- In a (nontrivial) complex Banach algebra, every element has nonempty spectrum. -/
protected theorem nonempty : (spectrum ℂ a).Nonempty := by
  /- Suppose `σ a = ∅`, then resolvent set is `ℂ`, any `(z • 1 - a)` is a unit, and `resolvent a`
    is differentiable on `ℂ`. -/
  /-
    A : Type u_2
    inst✝³ : NormedRing A
    inst✝² : NormedAlgebra Complex A
    inst✝¹ : CompleteSpace A
    inst✝ : Nontrivial A
    a : A
    ⊢ (spectrum Complex a).Nonempty
  -/
  by_contra! h
  /-
    A : Type u_2
    inst✝³ : NormedRing A
    inst✝² : NormedAlgebra Complex A
    inst✝¹ : CompleteSpace A
    inst✝ : Nontrivial A
    a : A
    h : Eq (spectrum Complex a) EmptyCollection.emptyCollection
    ⊢ False
  -/
  have H₀ : resolventSet ℂ a = Set.univ := by rwa [spectrum, Set.compl_empty_iff] at h
  have H₁ : Differentiable ℂ fun z : ℂ => resolvent a z := fun z =>
    (hasDerivAt_resolvent (H₀.symm ▸ Set.mem_univ z : z ∈ resolventSet ℂ a)).differentiableAt
  /- Since `resolvent a` tends to zero at infinity, by Liouville's theorem `resolvent a = 0`,
  which contradicts that `resolvent a z` is invertible. -/
  have H₃ := H₁.apply_eq_of_tendsto_cocompact 0 <| by
    simpa [Metric.cobounded_eq_cocompact] using resolvent_tendsto_cobounded a (𝕜 := ℂ)
  /-
    A : Type u_2
    inst✝³ : NormedRing A
    inst✝² : NormedAlgebra Complex A
    inst✝¹ : CompleteSpace A
    inst✝ : Nontrivial A
    a : A
    h : Eq (spectrum Complex a) EmptyCollection.emptyCollection
    H₀ : Eq (resolventSet Complex a) Set.univ
    H₁ : Differentiable Complex fun z => resolvent a z
    H₃ : Eq (resolvent a 0) 0
    ⊢ False
  -/
  exact not_isUnit_zero <| H₃ ▸ (isUnit_resolvent.mp <| H₀.symm ▸ Set.mem_univ 0)
  /-
    🎉 no goals
  -/


/-- In a complex Banach algebra, the spectral radius is always attained by some element of the
spectrum. -/
theorem exists_nnnorm_eq_spectralRadius : ∃ z ∈ spectrum ℂ a, (‖z‖₊ : ℝ≥0∞) = spectralRadius ℂ a :=
  exists_nnnorm_eq_spectralRadius_of_nonempty (spectrum.nonempty a)


/-- In a complex Banach algebra, if every element of the spectrum has norm strictly less than
`r : ℝ≥0`, then the spectral radius is also strictly less than `r`. -/
theorem spectralRadius_lt_of_forall_lt {r : ℝ≥0} (hr : ∀ z ∈ spectrum ℂ a, ‖z‖₊ < r) :
    spectralRadius ℂ a < r :=
  spectralRadius_lt_of_forall_lt_of_nonempty (spectrum.nonempty a) hr


/-- The **spectral mapping theorem** for polynomials in a Banach algebra over `ℂ`. -/
theorem map_polynomial_aeval (p : ℂ[X]) :
    spectrum ℂ (aeval a p) = (fun k => eval k p) '' spectrum ℂ a :=
  map_polynomial_aeval_of_nonempty a p (spectrum.nonempty a)


/-- A specialization of the spectral mapping theorem for polynomials in a Banach algebra over `ℂ`
to monic monomials. -/
protected theorem map_pow (n : ℕ) :
    spectrum ℂ (a ^ n) = (· ^ n) '' spectrum ℂ a := by
  /-
    A : Type u_2
    inst✝³ : NormedRing A
    inst✝² : NormedAlgebra Complex A
    inst✝¹ : CompleteSpace A
    inst✝ : Nontrivial A
    a : A
    n : Nat
    ⊢ Eq (spectrum Complex (HPow.hPow a n)) (Set.image (fun x => HPow.hPow x n) (s …
  -/
  simpa only [aeval_X_pow, eval_pow, eval_X] using map_polynomial_aeval a (X ^ n)
  /-
    🎉 no goals
  -/


local notation "σ" => spectrum ℂ


theorem algebraMap_eq_of_mem {a : A} {z : ℂ} (h : z ∈ σ a) : algebraMap ℂ A z = a := by
  /-
    A : Type u_2
    inst✝¹ : NormedRing A
    inst✝ : NormedAlgebra Complex A
    hA : ∀ {a : A}, Iff (IsUnit a) (Ne a 0)
    a : A
    z : Complex
    h : Membership.mem (spectrum Complex a) z
    ⊢ Eq ((algebraMap Complex A) z) a
  -/
  rwa [mem_iff, hA, Classical.not_not, sub_eq_zero] at h
  /-
    🎉 no goals
  -/


/-- **Gelfand-Mazur theorem**: For a complex Banach division algebra, the natural `algebraMap ℂ A`
is an algebra isomorphism whose inverse is given by selecting the (unique) element of
`spectrum ℂ a`. In addition, `algebraMap_isometry` guarantees this map is an isometry.

Note: because `NormedDivisionRing` requires the field `norm_mul' : ∀ a b, ‖a * b‖ = ‖a‖ * ‖b‖`, we
don't use this type class and instead opt for a `NormedRing` in which the nonzero elements are
precisely the units. This allows for the application of this isomorphism in broader contexts, e.g.,
to the quotient of a complex Banach algebra by a maximal ideal. In the case when `A` is actually a
`NormedDivisionRing`, one may fill in the argument `hA` with the lemma `isUnit_iff_ne_zero`. -/
@[simps]
noncomputable def _root_.NormedRing.algEquivComplexOfComplete [CompleteSpace A] : ℂ ≃ₐ[ℂ] A :=
  let nt : Nontrivial A := ⟨⟨1, 0, hA.mp ⟨⟨1, 1, mul_one _, mul_one _⟩, rfl⟩⟩⟩
  { Algebra.ofId ℂ A with
    toFun := algebraMap ℂ A
    invFun := fun a => (@spectrum.nonempty _ _ _ _ nt a).some
    left_inv := fun z => by
      simpa only [@scalar_eq _ _ _ _ _ nt _] using
        (@spectrum.nonempty _ _ _ _ nt <| algebraMap ℂ A z).some_mem
    right_inv := fun a => algebraMap_eq_of_mem (@hA) (@spectrum.nonempty _ _ _ _ nt a).some_mem }


/-- For `𝕜 = ℝ` or `𝕜 = ℂ`, `exp 𝕜` maps the spectrum of `a` into the spectrum of `exp 𝕜 a`. -/
theorem exp_mem_exp [RCLike 𝕜] [NormedRing A] [NormedAlgebra 𝕜 A] [CompleteSpace A] (a : A)
    {z : 𝕜} (hz : z ∈ spectrum 𝕜 a) : exp 𝕜 z ∈ spectrum 𝕜 (exp 𝕜 a) := by
  have hexpmul : exp 𝕜 a = exp 𝕜 (a - ↑ₐ z) * ↑ₐ (exp 𝕜 z) := by
    rw [algebraMap_exp_comm z, ← exp_add_of_commute (Algebra.commutes z (a - ↑ₐ z)).symm,
      sub_add_cancel]
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    z : 𝕜
    hz : Membership.mem (spectrum 𝕜 a) z
    hexpmul : Eq (NormedSpace.exp 𝕜 a) (HMul.hMul (NormedSpace.exp 𝕜 (HSub.hSub a  …
    ⊢ Membership.mem (spectrum 𝕜 (NormedSpace.exp 𝕜 a)) (NormedSpace.exp 𝕜 z)
  -/
  let b := ∑' n : ℕ, ((n + 1).factorial⁻¹ : 𝕜) • (a - ↑ₐ z) ^ n
  have hb : Summable fun n : ℕ => ((n + 1).factorial⁻¹ : 𝕜) • (a - ↑ₐ z) ^ n := by
    refine .of_norm_bounded_eventually _ (Real.summable_pow_div_factorial ‖a - ↑ₐ z‖) ?_
    filter_upwards [Filter.eventually_cofinite_ne 0] with n hn
    rw [norm_smul, mul_comm, norm_inv, RCLike.norm_natCast, ← div_eq_mul_inv]
    exact div_le_div₀ (pow_nonneg (norm_nonneg _) n) (norm_pow_le' (a - ↑ₐ z) (zero_lt_iff.mpr hn))
      (mod_cast Nat.factorial_pos n) (mod_cast Nat.factorial_le (lt_add_one n).le)
  have h₀ : (∑' n : ℕ, ((n + 1).factorial⁻¹ : 𝕜) • (a - ↑ₐ z) ^ (n + 1)) = (a - ↑ₐ z) * b := by
    simpa only [mul_smul_comm, pow_succ'] using hb.tsum_mul_left (a - ↑ₐ z)
  have h₁ : (∑' n : ℕ, ((n + 1).factorial⁻¹ : 𝕜) • (a - ↑ₐ z) ^ (n + 1)) = b * (a - ↑ₐ z) := by
    simpa only [pow_succ, Algebra.smul_mul_assoc] using hb.tsum_mul_right (a - ↑ₐ z)
  have h₃ : exp 𝕜 (a - ↑ₐ z) = 1 + (a - ↑ₐ z) * b := by
    rw [exp_eq_tsum]
    convert tsum_eq_zero_add (expSeries_summable' (𝕂 := 𝕜) (a - ↑ₐ z))
    · simp only [Nat.factorial_zero, Nat.cast_one, inv_one, pow_zero, one_smul]
    · exact h₀.symm
  rw [spectrum.mem_iff, IsUnit.sub_iff, ← one_mul (↑ₐ (exp 𝕜 z)), hexpmul, ← _root_.sub_mul,
    Commute.isUnit_mul_iff (Algebra.commutes (exp 𝕜 z) (exp 𝕜 (a - ↑ₐ z) - 1)).symm,
    sub_eq_iff_eq_add'.mpr h₃, Commute.isUnit_mul_iff (h₀ ▸ h₁ : (a - ↑ₐ z) * b = b * (a - ↑ₐ z))]
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : CompleteSpace A
    a : A
    z : 𝕜
    hz : Membership.mem (spectrum 𝕜 a) z
    hexpmul : Eq (NormedSpace.exp 𝕜 a) (HMul.hMul (NormedSpace.exp 𝕜 (HSub.hSub a  …
    b : A := tsum fun n => HSMul.hSMul (Inv.inv ↑(HAdd.hAdd n 1).factorial) (HPow. …
    hb : Summable fun n => HSMul.hSMul (Inv.inv ↑(HAdd.hAdd n 1).factorial) (HPow. …
    h₀ : Eq (tsum fun n => HSMul.hSMul (Inv.inv ↑(HAdd.hAdd n 1).factorial) (HPow. …
    h₁ : Eq (tsum fun n => HSMul.hSMul (Inv.inv ↑(HAdd.hAdd n 1).factorial) (HPow. …
    h₃ : Eq (NormedSpace.exp 𝕜 (HSub.hSub a ((algebraMap 𝕜 A) z))) (HAdd.hAdd 1 (H …
    ⊢ Not (And (And (IsUnit (HSub.hSub a ((algebraMap 𝕜 A) z))) (IsUnit b)) (IsUni …
  -/
  exact not_and_of_not_left _ (not_and_of_not_left _ ((not_iff_not.mpr IsUnit.sub_iff).mp hz))
  /-
    🎉 no goals
  -/


instance (priority := 100) [FunLike F A 𝕜] [AlgHomClass F 𝕜 A 𝕜] :
    ContinuousLinearMapClass F 𝕜 A 𝕜 :=
  { AlgHomClass.linearMapClass with
    map_continuous := fun φ =>
      AddMonoidHomClass.continuous_of_bound φ ‖(1 : A)‖ fun a =>
        mul_comm ‖a‖ ‖(1 : A)‖ ▸ spectrum.norm_le_norm_mul_of_mem (apply_mem_spectrum φ _) }


/-- An algebra homomorphism into the base field, as a continuous linear map (since it is
automatically bounded). -/
def toContinuousLinearMap (φ : A →ₐ[𝕜] 𝕜) : A →L[𝕜] 𝕜 :=
  { φ.toLinearMap with cont := map_continuous φ }


@[simp]
theorem coe_toContinuousLinearMap (φ : A →ₐ[𝕜] 𝕜) : ⇑φ.toContinuousLinearMap = φ :=
  rfl


theorem norm_apply_le_self_mul_norm_one [FunLike F A 𝕜] [AlgHomClass F 𝕜 A 𝕜] (f : F) (a : A) :
    ‖f a‖ ≤ ‖a‖ * ‖(1 : A)‖ :=
  spectrum.norm_le_norm_mul_of_mem (apply_mem_spectrum f _)


theorem norm_apply_le_self [NormOneClass A] [FunLike F A 𝕜] [AlgHomClass F 𝕜 A 𝕜]
    (f : F) (a : A) : ‖f a‖ ≤ ‖a‖ :=
  spectrum.norm_le_norm_of_mem (apply_mem_spectrum f _)


@[simp]
theorem toContinuousLinearMap_norm [NormOneClass A] (φ : A →ₐ[𝕜] 𝕜) :
    ‖φ.toContinuousLinearMap‖ = 1 :=
  ContinuousLinearMap.opNorm_eq_of_bounds zero_le_one
    (fun a => (one_mul ‖a‖).symm ▸ spectrum.norm_le_norm_of_mem (apply_mem_spectrum φ _))
                    /-
                      𝕜 : Type u_1
                      A : Type u_2
                      inst✝⁴ : NontriviallyNormedField 𝕜
                      inst✝³ : NormedRing A
                      inst✝² : NormedAlgebra 𝕜 A
                      inst✝¹ : CompleteSpace A
                      inst✝ : NormOneClass A
                      φ : AlgHom 𝕜 A 𝕜
                      x✝¹ : Real
                      x✝ : GE.ge x✝¹ 0
                      h : ∀ (x : A), LE.le (Norm.norm (φ.toContinuousLinearMap x)) (HMul.hMul x✝¹ (N …
                      ⊢ LE.le 1 x✝¹
                    -/
    fun _ _ h => by simpa only [coe_toContinuousLinearMap, map_one, norm_one, mul_one] using h 1
                    /-
                      🎉 no goals
                    -/


/-- The equivalence between characters and algebra homomorphisms into the base field. -/
def equivAlgHom : characterSpace 𝕜 A ≃ (A →ₐ[𝕜] 𝕜) where
  toFun := toAlgHom
  invFun f :=
    { val := f.toContinuousLinearMap
                     /-
                       𝕜 : Type u_1
                       A : Type u_2
                       inst✝³ : NontriviallyNormedField 𝕜
                       inst✝² : NormedRing A
                       inst✝¹ : CompleteSpace A
                       inst✝ : NormedAlgebra 𝕜 A
                       f : AlgHom 𝕜 A 𝕜
                       ⊢ Membership.mem (WeakDual.characterSpace 𝕜 A) f.toContinuousLinearMap
                     -/
      property := by rw [eq_set_map_one_map_mul]; exact ⟨map_one f, map_mul f⟩ }
                                                  /-
                                                    🎉 no goals
                                                  -/
  left_inv _ := Subtype.ext <| ContinuousLinearMap.ext fun _ => rfl
  right_inv _ := AlgHom.ext fun _ => rfl


@[simp]
theorem equivAlgHom_coe (f : characterSpace 𝕜 A) : ⇑(equivAlgHom f) = f :=
  rfl


@[simp]
theorem equivAlgHom_symm_coe (f : A →ₐ[𝕜] 𝕜) : ⇑(equivAlgHom.symm f) = f :=
  rfl


local notation "σ" => spectrum


open SubalgebraClass in
include instSMulMem in
/-- Let `S` be a closed subalgebra of a Banach algebra `A`. If `a : S` is invertible in `A`,
and for all `x : S` sufficiently close to `a` within some filter `l`, `x` is invertible in `S`,
then `a` is invertible in `S` as well. -/
lemma _root_.Subalgebra.isUnit_of_isUnit_val_of_eventually {l : Filter S} {a : S}
    (ha : IsUnit (a : A)) (hla : l ≤ 𝓝 a) (hl : ∀ᶠ x in l, IsUnit x) (hl' : l.NeBot) :
    IsUnit a := by
  have hla₂ : Tendsto Ring.inverse (map (val S) l) (𝓝 (↑ha.unit⁻¹ : A)) := by
    rw [← Ring.inverse_unit]
    exact (NormedRing.inverse_continuousAt _).tendsto.comp <|
      continuousAt_subtype_val.tendsto.comp <| map_mono hla
  suffices mem : (↑ha.unit⁻¹ : A) ∈ S by
    refine ⟨⟨a, ⟨(↑ha.unit⁻¹ : A), mem⟩, ?_, ?_⟩, rfl⟩
    all_goals ext; simp
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    l : Filter (Subtype fun x => Membership.mem S x)
    a : Subtype fun x => Membership.mem S x
    ha : IsUnit ↑a
    hla : LE.le l (nhds a)
    hl : Filter.Eventually (fun x => IsUnit x) l
    hl' : l.NeBot
    hla₂ : Filter.Tendsto Ring.inverse (Filter.map (⇑(SubalgebraClass.val S)) l) ( …
    ⊢ Membership.mem S ↑(Inv.inv ha.unit)
  -/
  apply hS.mem_of_tendsto hla₂
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    l : Filter (Subtype fun x => Membership.mem S x)
    a : Subtype fun x => Membership.mem S x
    ha : IsUnit ↑a
    hla : LE.le l (nhds a)
    hl : Filter.Eventually (fun x => IsUnit x) l
    hl' : l.NeBot
    hla₂ : Filter.Tendsto Ring.inverse (Filter.map (⇑(SubalgebraClass.val S)) l) ( …
    ⊢ Filter.Eventually (fun x => Membership.mem (↑S) (Ring.inverse x)) (Filter.ma …
  -/
  rw [Filter.eventually_map]
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    l : Filter (Subtype fun x => Membership.mem S x)
    a : Subtype fun x => Membership.mem S x
    ha : IsUnit ↑a
    hla : LE.le l (nhds a)
    hl : Filter.Eventually (fun x => IsUnit x) l
    hl' : l.NeBot
    hla₂ : Filter.Tendsto Ring.inverse (Filter.map (⇑(SubalgebraClass.val S)) l) ( …
    ⊢ Filter.Eventually (fun a => Membership.mem (↑S) (Ring.inverse ((SubalgebraCl …
  -/
  apply hl.mono fun x hx ↦ ?_
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    l : Filter (Subtype fun x => Membership.mem S x)
    a : Subtype fun x => Membership.mem S x
    ha : IsUnit ↑a
    hla : LE.le l (nhds a)
    hl : Filter.Eventually (fun x => IsUnit x) l
    hl' : l.NeBot
    hla₂ : Filter.Tendsto Ring.inverse (Filter.map (⇑(SubalgebraClass.val S)) l) ( …
    x : Subtype fun x => Membership.mem S x
    hx : IsUnit x
    ⊢ Membership.mem (↑S) (Ring.inverse ((SubalgebraClass.val S) x))
  -/
  suffices Ring.inverse (val S x) = (val S ↑hx.unit⁻¹) from this ▸ Subtype.property _
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    l : Filter (Subtype fun x => Membership.mem S x)
    a : Subtype fun x => Membership.mem S x
    ha : IsUnit ↑a
    hla : LE.le l (nhds a)
    hl : Filter.Eventually (fun x => IsUnit x) l
    hl' : l.NeBot
    hla₂ : Filter.Tendsto Ring.inverse (Filter.map (⇑(SubalgebraClass.val S)) l) ( …
    x : Subtype fun x => Membership.mem S x
    hx : IsUnit x
    ⊢ Eq (Ring.inverse ((SubalgebraClass.val S) x)) ((SubalgebraClass.val S) ↑(Inv …
  -/
  rw [← (hx.map (val S)).unit_spec, Ring.inverse_unit (hx.map (val S)).unit, val]
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    l : Filter (Subtype fun x => Membership.mem S x)
    a : Subtype fun x => Membership.mem S x
    ha : IsUnit ↑a
    hla : LE.le l (nhds a)
    hl : Filter.Eventually (fun x => IsUnit x) l
    hl' : l.NeBot
    hla₂ : Filter.Tendsto Ring.inverse (Filter.map (⇑(SubalgebraClass.val S)) l) ( …
    x : Subtype fun x => Membership.mem S x
    hx : IsUnit x
    ⊢ Eq (↑(Inv.inv ⋯.unit)) ({ toFun := Subtype.val, map_one' := ⋯, map_mul' := ⋯ …
  -/
  apply Units.mul_eq_one_iff_inv_eq.mp
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    l : Filter (Subtype fun x => Membership.mem S x)
    a : Subtype fun x => Membership.mem S x
    ha : IsUnit ↑a
    hla : LE.le l (nhds a)
    hl : Filter.Eventually (fun x => IsUnit x) l
    hl' : l.NeBot
    hla₂ : Filter.Tendsto Ring.inverse (Filter.map (⇑(SubalgebraClass.val S)) l) ( …
    x : Subtype fun x => Membership.mem S x
    hx : IsUnit x
    ⊢ Eq (HMul.hMul (↑⋯.unit) ({ toFun := Subtype.val, map_one' := ⋯, map_mul' :=  …
  -/
  simpa [-IsUnit.mul_val_inv] using congr(($hx.mul_val_inv : A))
  /-
    🎉 no goals
  -/


/-- If `S : Subalgebra 𝕜 A` is a closed subalgebra of a Banach algebra `A`, then for any
`x : S`, the boundary of the spectrum of `x` relative to `S` is a subset of the spectrum of
`↑x : A` relative to `A`. -/
lemma _root_.Subalgebra.frontier_spectrum : frontier (σ 𝕜 x) ⊆ σ 𝕜 (x : A) := by
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    ⊢ HasSubset.Subset (frontier (spectrum 𝕜 x)) (spectrum 𝕜 ↑x)
  -/
  have : CompleteSpace S := hS.completeSpace_coe
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    this : CompleteSpace (Subtype fun x => Membership.mem S x)
    ⊢ HasSubset.Subset (frontier (spectrum 𝕜 x)) (spectrum 𝕜 ↑x)
  -/
  intro μ hμ
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    this : CompleteSpace (Subtype fun x => Membership.mem S x)
    μ : 𝕜
    hμ : Membership.mem (frontier (spectrum 𝕜 x)) μ
    ⊢ Membership.mem (spectrum 𝕜 ↑x) μ
  -/
  by_contra h
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    this : CompleteSpace (Subtype fun x => Membership.mem S x)
    μ : 𝕜
    hμ : Membership.mem (frontier (spectrum 𝕜 x)) μ
    h : Not (Membership.mem (spectrum 𝕜 ↑x) μ)
    ⊢ False
  -/
  rw [spectrum.not_mem_iff] at h
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    this : CompleteSpace (Subtype fun x => Membership.mem S x)
    μ : 𝕜
    hμ : Membership.mem (frontier (spectrum 𝕜 x)) μ
    h : IsUnit (HSub.hSub ((algebraMap 𝕜 A) μ) ↑x)
    ⊢ False
  -/
  rw [← frontier_compl, (spectrum.isClosed _).isOpen_compl.frontier_eq, mem_diff] at hμ
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    this : CompleteSpace (Subtype fun x => Membership.mem S x)
    μ : 𝕜
    hμ : And (Membership.mem (closure (HasCompl.compl (spectrum 𝕜 x))) μ) (Not (Me …
    h : IsUnit (HSub.hSub ((algebraMap 𝕜 A) μ) ↑x)
    ⊢ False
  -/
  obtain ⟨hμ₁, hμ₂⟩ := hμ
  /-
    case intro
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    this : CompleteSpace (Subtype fun x => Membership.mem S x)
    μ : 𝕜
    h : IsUnit (HSub.hSub ((algebraMap 𝕜 A) μ) ↑x)
    hμ₁ : Membership.mem (closure (HasCompl.compl (spectrum 𝕜 x))) μ
    hμ₂ : Not (Membership.mem (HasCompl.compl (spectrum 𝕜 x)) μ)
    ⊢ False
  -/
  rw [mem_closure_iff_clusterPt] at hμ₁
  /-
    case intro
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    this : CompleteSpace (Subtype fun x => Membership.mem S x)
    μ : 𝕜
    h : IsUnit (HSub.hSub ((algebraMap 𝕜 A) μ) ↑x)
    hμ₁ : ClusterPt μ (Filter.principal (HasCompl.compl (spectrum 𝕜 x)))
    hμ₂ : Not (Membership.mem (HasCompl.compl (spectrum 𝕜 x)) μ)
    ⊢ False
  -/
  apply hμ₂
  /-
    case intro
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    this : CompleteSpace (Subtype fun x => Membership.mem S x)
    μ : 𝕜
    h : IsUnit (HSub.hSub ((algebraMap 𝕜 A) μ) ↑x)
    hμ₁ : ClusterPt μ (Filter.principal (HasCompl.compl (spectrum 𝕜 x)))
    hμ₂ : Not (Membership.mem (HasCompl.compl (spectrum 𝕜 x)) μ)
    ⊢ Membership.mem (HasCompl.compl (spectrum 𝕜 x)) μ
  -/
  rw [mem_compl_iff, spectrum.not_mem_iff]
  /-
    case intro
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    this : CompleteSpace (Subtype fun x => Membership.mem S x)
    μ : 𝕜
    h : IsUnit (HSub.hSub ((algebraMap 𝕜 A) μ) ↑x)
    hμ₁ : ClusterPt μ (Filter.principal (HasCompl.compl (spectrum 𝕜 x)))
    hμ₂ : Not (Membership.mem (HasCompl.compl (spectrum 𝕜 x)) μ)
    ⊢ IsUnit (HSub.hSub ((algebraMap 𝕜 (Subtype fun x => Membership.mem S x)) μ) x)
  -/
  refine Subalgebra.isUnit_of_isUnit_val_of_eventually S h ?_ ?_ <| .map hμ₁ (algebraMap 𝕜 S · - x)
  · calc
      _ ≤ map _ (𝓝 μ) := map_mono (by simp)
      _ ≤ _ := by rw [← Filter.Tendsto, ← ContinuousAt]; fun_prop
    /-
      case intro.refine_2
      𝕜 : Type u_3
      A : Type u_4
      SA : Type u_5
      inst✝⁵ : NormedRing A
      inst✝⁴ : CompleteSpace A
      inst✝³ : SetLike SA A
      inst✝² : SubringClass SA A
      inst✝¹ : NormedField 𝕜
      inst✝ : NormedAlgebra 𝕜 A
      instSMulMem : SMulMemClass SA 𝕜 A
      S : SA
      hS : IsClosed ↑S
      x : Subtype fun x => Membership.mem S x
      this : CompleteSpace (Subtype fun x => Membership.mem S x)
      μ : 𝕜
      h : IsUnit (HSub.hSub ((algebraMap 𝕜 A) μ) ↑x)
      hμ₁ : ClusterPt μ (Filter.principal (HasCompl.compl (spectrum 𝕜 x)))
      hμ₂ : Not (Membership.mem (HasCompl.compl (spectrum 𝕜 x)) μ)
      ⊢ Filter.Eventually (fun x => IsUnit x) (Filter.map (fun x_1 => HSub.hSub ((al …
    -/
  · rw [eventually_map]
    /-
      case intro.refine_2
      𝕜 : Type u_3
      A : Type u_4
      SA : Type u_5
      inst✝⁵ : NormedRing A
      inst✝⁴ : CompleteSpace A
      inst✝³ : SetLike SA A
      inst✝² : SubringClass SA A
      inst✝¹ : NormedField 𝕜
      inst✝ : NormedAlgebra 𝕜 A
      instSMulMem : SMulMemClass SA 𝕜 A
      S : SA
      hS : IsClosed ↑S
      x : Subtype fun x => Membership.mem S x
      this : CompleteSpace (Subtype fun x => Membership.mem S x)
      μ : 𝕜
      h : IsUnit (HSub.hSub ((algebraMap 𝕜 A) μ) ↑x)
      hμ₁ : ClusterPt μ (Filter.principal (HasCompl.compl (spectrum 𝕜 x)))
      hμ₂ : Not (Membership.mem (HasCompl.compl (spectrum 𝕜 x)) μ)
      ⊢ Filter.Eventually (fun a => IsUnit (HSub.hSub ((algebraMap 𝕜 (Subtype fun x  …
    -/
    apply Eventually.filter_mono inf_le_right
    /-
      case intro.refine_2
      𝕜 : Type u_3
      A : Type u_4
      SA : Type u_5
      inst✝⁵ : NormedRing A
      inst✝⁴ : CompleteSpace A
      inst✝³ : SetLike SA A
      inst✝² : SubringClass SA A
      inst✝¹ : NormedField 𝕜
      inst✝ : NormedAlgebra 𝕜 A
      instSMulMem : SMulMemClass SA 𝕜 A
      S : SA
      hS : IsClosed ↑S
      x : Subtype fun x => Membership.mem S x
      this : CompleteSpace (Subtype fun x => Membership.mem S x)
      μ : 𝕜
      h : IsUnit (HSub.hSub ((algebraMap 𝕜 A) μ) ↑x)
      hμ₁ : ClusterPt μ (Filter.principal (HasCompl.compl (spectrum 𝕜 x)))
      hμ₂ : Not (Membership.mem (HasCompl.compl (spectrum 𝕜 x)) μ)
      ⊢ Filter.Eventually (fun x_1 => IsUnit (HSub.hSub ((algebraMap 𝕜 (Subtype fun  …
    -/
    simp [spectrum.not_mem_iff]
    /-
      🎉 no goals
    -/


/-- If `S` is a closed subalgebra of a Banach algebra `A`, then for any `x : S`, the boundary of
the spectrum of `x` relative to `S` is a subset of the boundary of the spectrum of `↑x : A`
relative to `A`. -/
lemma Subalgebra.frontier_subset_frontier :
    frontier (σ 𝕜 x) ⊆ frontier (σ 𝕜 (x : A)) := by
  rw [frontier_eq_closure_inter_closure (s := σ 𝕜 (x : A)),
    (spectrum.isClosed (x : A)).closure_eq]
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    ⊢ HasSubset.Subset (frontier (spectrum 𝕜 x)) (Inter.inter (spectrum 𝕜 ↑x) (clo …
  -/
  apply subset_inter (frontier_spectrum S x)
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    ⊢ HasSubset.Subset (frontier (spectrum 𝕜 x)) (closure (HasCompl.compl (spectru …
  -/
  rw [frontier_eq_closure_inter_closure]
  exact inter_subset_right |>.trans <|
    closure_mono <| compl_subset_compl.mpr <| spectrum.subset_subalgebra x


/-- If `S` is a closed subalgebra of a Banach algebra `A`, then for any `x : S`, the spectrum of `x`
is the spectrum of `↑x : A` along with the connected components of the complement of the spectrum of
`↑x : A` which contain an element of the spectrum of `x : S`. -/
lemma Subalgebra.spectrum_sUnion_connectedComponentIn :
    σ 𝕜 x = σ 𝕜 (x : A) ∪ (⋃ z ∈ (σ 𝕜 x \ σ 𝕜 (x : A)), connectedComponentIn (σ 𝕜 (x : A))ᶜ z) := by
  suffices IsClopen ((σ 𝕜 (x : A))ᶜ ↓∩ (σ 𝕜 x \ σ 𝕜 (x : A))) by
    rw [← this.biUnion_connectedComponentIn (diff_subset_compl _ _),
      union_diff_cancel (spectrum.subset_subalgebra x)]
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    ⊢ IsClopen (Set.preimage Subtype.val (SDiff.sdiff (spectrum 𝕜 x) (spectrum 𝕜 ↑ …
  -/
  have : CompleteSpace S := hS.completeSpace_coe
  have h_open : IsOpen (σ 𝕜 x \ σ 𝕜 (x : A)) := by
    rw [← (spectrum.isClosed (𝕜 := 𝕜) x).closure_eq, closure_eq_interior_union_frontier,
      union_diff_distrib, diff_eq_empty.mpr (frontier_spectrum S x),
      diff_eq_compl_inter, union_empty]
    exact (spectrum.isClosed _).isOpen_compl.inter isOpen_interior
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    this : CompleteSpace (Subtype fun x => Membership.mem S x)
    h_open : IsOpen (SDiff.sdiff (spectrum 𝕜 x) (spectrum 𝕜 ↑x))
    ⊢ IsClopen (Set.preimage Subtype.val (SDiff.sdiff (spectrum 𝕜 x) (spectrum 𝕜 ↑ …
  -/
  apply isClopen_preimage_val h_open
  suffices h_frontier : frontier (σ 𝕜 x \ σ 𝕜 (x : A)) ⊆ frontier (σ 𝕜 (x : A)) from
    disjoint_of_subset_left h_frontier <| disjoint_compl_right.frontier_left
      (spectrum.isClosed _).isOpen_compl
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    this : CompleteSpace (Subtype fun x => Membership.mem S x)
    h_open : IsOpen (SDiff.sdiff (spectrum 𝕜 x) (spectrum 𝕜 ↑x))
    ⊢ HasSubset.Subset (frontier (SDiff.sdiff (spectrum 𝕜 x) (spectrum 𝕜 ↑x))) (fr …
  -/
  rw [diff_eq_compl_inter]
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    this : CompleteSpace (Subtype fun x => Membership.mem S x)
    h_open : IsOpen (SDiff.sdiff (spectrum 𝕜 x) (spectrum 𝕜 ↑x))
    ⊢ HasSubset.Subset (frontier (Inter.inter (HasCompl.compl (spectrum 𝕜 ↑x)) (sp …
  -/
  apply (frontier_inter_subset _ _).trans
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    this : CompleteSpace (Subtype fun x => Membership.mem S x)
    h_open : IsOpen (SDiff.sdiff (spectrum 𝕜 x) (spectrum 𝕜 ↑x))
    ⊢ HasSubset.Subset (Union.union (Inter.inter (frontier (HasCompl.compl (spectr …
  -/
  rw [frontier_compl]
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    this : CompleteSpace (Subtype fun x => Membership.mem S x)
    h_open : IsOpen (SDiff.sdiff (spectrum 𝕜 x) (spectrum 𝕜 ↑x))
    ⊢ HasSubset.Subset (Union.union (Inter.inter (frontier (spectrum 𝕜 ↑x)) (closu …
  -/
  apply union_subset <| inter_subset_left
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    this : CompleteSpace (Subtype fun x => Membership.mem S x)
    h_open : IsOpen (SDiff.sdiff (spectrum 𝕜 x) (spectrum 𝕜 ↑x))
    ⊢ HasSubset.Subset (Inter.inter (closure (HasCompl.compl (spectrum 𝕜 ↑x))) (fr …
  -/
  refine inter_subset_inter_right _ ?_ |>.trans <| inter_subset_right
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    this : CompleteSpace (Subtype fun x => Membership.mem S x)
    h_open : IsOpen (SDiff.sdiff (spectrum 𝕜 x) (spectrum 𝕜 ↑x))
    ⊢ HasSubset.Subset (frontier (spectrum 𝕜 x)) (frontier (spectrum 𝕜 ↑x))
  -/
  exact frontier_subset_frontier S x
  /-
    🎉 no goals
  -/


/-- Let `S` be a closed subalgebra of a Banach algebra `A`, and let `x : S`. If `z` is in the
spectrum of `x`, then the connected component of `z` in the complement of the spectrum of `↑x : A`
is bounded (or else `z` actually belongs to the spectrum of `↑x : A`). -/
lemma Subalgebra.spectrum_isBounded_connectedComponentIn {z : 𝕜} (hz : z ∈ σ 𝕜 x) :
    Bornology.IsBounded (connectedComponentIn (σ 𝕜 (x : A))ᶜ z) := by
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁵ : NormedRing A
    inst✝⁴ : CompleteSpace A
    inst✝³ : SetLike SA A
    inst✝² : SubringClass SA A
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedAlgebra 𝕜 A
    instSMulMem : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    z : 𝕜
    hz : Membership.mem (spectrum 𝕜 x) z
    ⊢ Bornology.IsBounded (connectedComponentIn (HasCompl.compl (spectrum 𝕜 ↑x)) z)
  -/
  by_cases hz' : z ∈ σ 𝕜 (x : A)
    /-
      case pos
      𝕜 : Type u_3
      A : Type u_4
      SA : Type u_5
      inst✝⁵ : NormedRing A
      inst✝⁴ : CompleteSpace A
      inst✝³ : SetLike SA A
      inst✝² : SubringClass SA A
      inst✝¹ : NormedField 𝕜
      inst✝ : NormedAlgebra 𝕜 A
      instSMulMem : SMulMemClass SA 𝕜 A
      S : SA
      hS : IsClosed ↑S
      x : Subtype fun x => Membership.mem S x
      z : 𝕜
      hz : Membership.mem (spectrum 𝕜 x) z
      hz' : Membership.mem (spectrum 𝕜 ↑x) z
      ⊢ Bornology.IsBounded (connectedComponentIn (HasCompl.compl (spectrum 𝕜 ↑x)) z)
    -/
  · simp [connectedComponentIn_eq_empty (show z ∉ (σ 𝕜 (x : A))ᶜ from not_not.mpr hz')]
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_3
      A : Type u_4
      SA : Type u_5
      inst✝⁵ : NormedRing A
      inst✝⁴ : CompleteSpace A
      inst✝³ : SetLike SA A
      inst✝² : SubringClass SA A
      inst✝¹ : NormedField 𝕜
      inst✝ : NormedAlgebra 𝕜 A
      instSMulMem : SMulMemClass SA 𝕜 A
      S : SA
      hS : IsClosed ↑S
      x : Subtype fun x => Membership.mem S x
      z : 𝕜
      hz : Membership.mem (spectrum 𝕜 x) z
      hz' : Not (Membership.mem (spectrum 𝕜 ↑x) z)
      ⊢ Bornology.IsBounded (connectedComponentIn (HasCompl.compl (spectrum 𝕜 ↑x)) z)
    -/
  · have : CompleteSpace S := hS.completeSpace_coe
    /-
      case neg
      𝕜 : Type u_3
      A : Type u_4
      SA : Type u_5
      inst✝⁵ : NormedRing A
      inst✝⁴ : CompleteSpace A
      inst✝³ : SetLike SA A
      inst✝² : SubringClass SA A
      inst✝¹ : NormedField 𝕜
      inst✝ : NormedAlgebra 𝕜 A
      instSMulMem : SMulMemClass SA 𝕜 A
      S : SA
      hS : IsClosed ↑S
      x : Subtype fun x => Membership.mem S x
      z : 𝕜
      hz : Membership.mem (spectrum 𝕜 x) z
      hz' : Not (Membership.mem (spectrum 𝕜 ↑x) z)
      this : CompleteSpace (Subtype fun x => Membership.mem S x)
      ⊢ Bornology.IsBounded (connectedComponentIn (HasCompl.compl (spectrum 𝕜 ↑x)) z)
    -/
    suffices connectedComponentIn (σ 𝕜 (x : A))ᶜ z ⊆ σ 𝕜 x from spectrum.isBounded x |>.subset this
    /-
      case neg
      𝕜 : Type u_3
      A : Type u_4
      SA : Type u_5
      inst✝⁵ : NormedRing A
      inst✝⁴ : CompleteSpace A
      inst✝³ : SetLike SA A
      inst✝² : SubringClass SA A
      inst✝¹ : NormedField 𝕜
      inst✝ : NormedAlgebra 𝕜 A
      instSMulMem : SMulMemClass SA 𝕜 A
      S : SA
      hS : IsClosed ↑S
      x : Subtype fun x => Membership.mem S x
      z : 𝕜
      hz : Membership.mem (spectrum 𝕜 x) z
      hz' : Not (Membership.mem (spectrum 𝕜 ↑x) z)
      this : CompleteSpace (Subtype fun x => Membership.mem S x)
      ⊢ HasSubset.Subset (connectedComponentIn (HasCompl.compl (spectrum 𝕜 ↑x)) z) ( …
    -/
    rw [spectrum_sUnion_connectedComponentIn S]
    /-
      case neg
      𝕜 : Type u_3
      A : Type u_4
      SA : Type u_5
      inst✝⁵ : NormedRing A
      inst✝⁴ : CompleteSpace A
      inst✝³ : SetLike SA A
      inst✝² : SubringClass SA A
      inst✝¹ : NormedField 𝕜
      inst✝ : NormedAlgebra 𝕜 A
      instSMulMem : SMulMemClass SA 𝕜 A
      S : SA
      hS : IsClosed ↑S
      x : Subtype fun x => Membership.mem S x
      z : 𝕜
      hz : Membership.mem (spectrum 𝕜 x) z
      hz' : Not (Membership.mem (spectrum 𝕜 ↑x) z)
      this : CompleteSpace (Subtype fun x => Membership.mem S x)
      ⊢ HasSubset.Subset (connectedComponentIn (HasCompl.compl (spectrum 𝕜 ↑x)) z) ( …
    -/
    exact subset_biUnion_of_mem (mem_diff_of_mem hz hz') |>.trans subset_union_right
    /-
      🎉 no goals
    -/


/-- Let `S` be a closed subalgebra of a Banach algebra `A`. If for `x : S` the complement of the
spectrum of `↑x : A` is connected, then `spectrum 𝕜 x = spectrum 𝕜 (x : A)`. -/
lemma Subalgebra.spectrum_eq_of_isPreconnected_compl (h : IsPreconnected (σ 𝕜 (x : A))ᶜ) :
    σ 𝕜 x = σ 𝕜 (x : A) := by
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁶ : NormedRing A
    inst✝⁵ : CompleteSpace A
    inst✝⁴ : SetLike SA A
    inst✝³ : SubringClass SA A
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    h : IsPreconnected (HasCompl.compl (spectrum 𝕜 ↑x))
    ⊢ Eq (spectrum 𝕜 x) (spectrum 𝕜 ↑x)
  -/
  nontriviality A
  suffices σ 𝕜 x \ σ 𝕜 (x : A) = ∅ by
    rw [spectrum_sUnion_connectedComponentIn, this]
    simp
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁶ : NormedRing A
    inst✝⁵ : CompleteSpace A
    inst✝⁴ : SetLike SA A
    inst✝³ : SubringClass SA A
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    h : IsPreconnected (HasCompl.compl (spectrum 𝕜 ↑x))
    a✝ : Nontrivial A
    ⊢ Eq (SDiff.sdiff (spectrum 𝕜 x) (spectrum 𝕜 ↑x)) EmptyCollection.emptyCollect …
  -/
  refine eq_empty_of_forall_not_mem fun z hz ↦ NormedSpace.unbounded_univ 𝕜 𝕜 ?_
  /-
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁶ : NormedRing A
    inst✝⁵ : CompleteSpace A
    inst✝⁴ : SetLike SA A
    inst✝³ : SubringClass SA A
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    h : IsPreconnected (HasCompl.compl (spectrum 𝕜 ↑x))
    a✝ : Nontrivial A
    z : 𝕜
    hz : Membership.mem (SDiff.sdiff (spectrum 𝕜 x) (spectrum 𝕜 ↑x)) z
    ⊢ Bornology.IsBounded Set.univ
  -/
  obtain ⟨hz, hz'⟩ := mem_diff _ |>.mp hz
  have := (spectrum.isBounded (x : A)).union <|
    h.connectedComponentIn hz' ▸ spectrum_isBounded_connectedComponentIn S x hz
  /-
    case intro
    𝕜 : Type u_3
    A : Type u_4
    SA : Type u_5
    inst✝⁶ : NormedRing A
    inst✝⁵ : CompleteSpace A
    inst✝⁴ : SetLike SA A
    inst✝³ : SubringClass SA A
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : SMulMemClass SA 𝕜 A
    S : SA
    hS : IsClosed ↑S
    x : Subtype fun x => Membership.mem S x
    h : IsPreconnected (HasCompl.compl (spectrum 𝕜 ↑x))
    a✝ : Nontrivial A
    z : 𝕜
    hz✝ : Membership.mem (SDiff.sdiff (spectrum 𝕜 x) (spectrum 𝕜 ↑x)) z
    hz : Membership.mem (spectrum 𝕜 x) z
    hz' : Not (Membership.mem (spectrum 𝕜 ↑x) z)
    this : Bornology.IsBounded (Union.union (spectrum 𝕜 ↑x) (HasCompl.compl (spect …
    ⊢ Bornology.IsBounded Set.univ
  -/
  simpa
  /-
    🎉 no goals
  -/


/-- If `𝕜₁` is a normed field contained as subfield of a larger normed field `𝕜₂`, and if `a : A`
is an element whose `𝕜₂` spectrum restricts to `𝕜₁`, then the spectral radii over each scalar
field coincide. -/
lemma spectralRadius_eq {𝕜₁ 𝕜₂ A : Type*} [NormedField 𝕜₁] [NormedField 𝕜₂]
    [NormedRing A] [NormedAlgebra 𝕜₁ A] [NormedAlgebra 𝕜₂ A] [NormedAlgebra 𝕜₁ 𝕜₂]
    [IsScalarTower 𝕜₁ 𝕜₂ A] {f : 𝕜₂ → 𝕜₁} {a : A} (h : SpectrumRestricts a f) :
    spectralRadius 𝕜₁ a = spectralRadius 𝕜₂ a := by
  /-
    𝕜₁ : Type u_3
    𝕜₂ : Type u_4
    A : Type u_5
    inst✝⁶ : NormedField 𝕜₁
    inst✝⁵ : NormedField 𝕜₂
    inst✝⁴ : NormedRing A
    inst✝³ : NormedAlgebra 𝕜₁ A
    inst✝² : NormedAlgebra 𝕜₂ A
    inst✝¹ : NormedAlgebra 𝕜₁ 𝕜₂
    inst✝ : IsScalarTower 𝕜₁ 𝕜₂ A
    f : 𝕜₂ → 𝕜₁
    a : A
    h : SpectrumRestricts a f
    ⊢ Eq (spectralRadius 𝕜₁ a) (spectralRadius 𝕜₂ a)
  -/
  rw [spectralRadius, spectralRadius]
  /-
    𝕜₁ : Type u_3
    𝕜₂ : Type u_4
    A : Type u_5
    inst✝⁶ : NormedField 𝕜₁
    inst✝⁵ : NormedField 𝕜₂
    inst✝⁴ : NormedRing A
    inst✝³ : NormedAlgebra 𝕜₁ A
    inst✝² : NormedAlgebra 𝕜₂ A
    inst✝¹ : NormedAlgebra 𝕜₁ 𝕜₂
    inst✝ : IsScalarTower 𝕜₁ 𝕜₂ A
    f : 𝕜₂ → 𝕜₁
    a : A
    h : SpectrumRestricts a f
    ⊢ Eq (iSup fun k => iSup fun h => ↑(NNNorm.nnnorm k)) (iSup fun k => iSup fun  …
  -/
  have := algebraMap_isometry 𝕜₁ 𝕜₂ |>.nnnorm_map_of_map_zero (map_zero _)
  /-
    𝕜₁ : Type u_3
    𝕜₂ : Type u_4
    A : Type u_5
    inst✝⁶ : NormedField 𝕜₁
    inst✝⁵ : NormedField 𝕜₂
    inst✝⁴ : NormedRing A
    inst✝³ : NormedAlgebra 𝕜₁ A
    inst✝² : NormedAlgebra 𝕜₂ A
    inst✝¹ : NormedAlgebra 𝕜₁ 𝕜₂
    inst✝ : IsScalarTower 𝕜₁ 𝕜₂ A
    f : 𝕜₂ → 𝕜₁
    a : A
    h : SpectrumRestricts a f
    this : ∀ (x : 𝕜₁), Eq (NNNorm.nnnorm ((algebraMap 𝕜₁ 𝕜₂) x)) (NNNorm.nnnorm x)
    ⊢ Eq (iSup fun k => iSup fun h => ↑(NNNorm.nnnorm k)) (iSup fun k => iSup fun  …
  -/
  apply le_antisymm
  /-
    case a
    𝕜₁ : Type u_3
    𝕜₂ : Type u_4
    A : Type u_5
    inst✝⁶ : NormedField 𝕜₁
    inst✝⁵ : NormedField 𝕜₂
    inst✝⁴ : NormedRing A
    inst✝³ : NormedAlgebra 𝕜₁ A
    inst✝² : NormedAlgebra 𝕜₂ A
    inst✝¹ : NormedAlgebra 𝕜₁ 𝕜₂
    inst✝ : IsScalarTower 𝕜₁ 𝕜₂ A
    f : 𝕜₂ → 𝕜₁
    a : A
    h : SpectrumRestricts a f
    this : ∀ (x : 𝕜₁), Eq (NNNorm.nnnorm ((algebraMap 𝕜₁ 𝕜₂) x)) (NNNorm.nnnorm x)
    ⊢ LE.le (iSup fun k => iSup fun h => ↑(NNNorm.nnnorm k)) (iSup fun k => iSup f …
  -/
  all_goals apply iSup₂_le fun x hx ↦ ?_
    /-
      𝕜₁ : Type u_3
      𝕜₂ : Type u_4
      A : Type u_5
      inst✝⁶ : NormedField 𝕜₁
      inst✝⁵ : NormedField 𝕜₂
      inst✝⁴ : NormedRing A
      inst✝³ : NormedAlgebra 𝕜₁ A
      inst✝² : NormedAlgebra 𝕜₂ A
      inst✝¹ : NormedAlgebra 𝕜₁ 𝕜₂
      inst✝ : IsScalarTower 𝕜₁ 𝕜₂ A
      f : 𝕜₂ → 𝕜₁
      a : A
      h : SpectrumRestricts a f
      this : ∀ (x : 𝕜₁), Eq (NNNorm.nnnorm ((algebraMap 𝕜₁ 𝕜₂) x)) (NNNorm.nnnorm x)
      x : 𝕜₁
      hx : Membership.mem (spectrum 𝕜₁ a) x
      ⊢ LE.le (↑(NNNorm.nnnorm x)) (iSup fun k => iSup fun h => ↑(NNNorm.nnnorm k))
    -/
  · refine congr_arg ((↑) : ℝ≥0 → ℝ≥0∞) (this x) |>.symm.trans_le <| le_iSup₂ (α := ℝ≥0∞) _ ?_
    /-
      𝕜₁ : Type u_3
      𝕜₂ : Type u_4
      A : Type u_5
      inst✝⁶ : NormedField 𝕜₁
      inst✝⁵ : NormedField 𝕜₂
      inst✝⁴ : NormedRing A
      inst✝³ : NormedAlgebra 𝕜₁ A
      inst✝² : NormedAlgebra 𝕜₂ A
      inst✝¹ : NormedAlgebra 𝕜₁ 𝕜₂
      inst✝ : IsScalarTower 𝕜₁ 𝕜₂ A
      f : 𝕜₂ → 𝕜₁
      a : A
      h : SpectrumRestricts a f
      this : ∀ (x : 𝕜₁), Eq (NNNorm.nnnorm ((algebraMap 𝕜₁ 𝕜₂) x)) (NNNorm.nnnorm x)
      x : 𝕜₁
      hx : Membership.mem (spectrum 𝕜₁ a) x
      ⊢ Membership.mem (spectrum 𝕜₂ a) ((algebraMap 𝕜₁ 𝕜₂) x)
    -/
    exact (spectrum.algebraMap_mem_iff _).mpr hx
    /-
      🎉 no goals
    -/
    /-
      𝕜₁ : Type u_3
      𝕜₂ : Type u_4
      A : Type u_5
      inst✝⁶ : NormedField 𝕜₁
      inst✝⁵ : NormedField 𝕜₂
      inst✝⁴ : NormedRing A
      inst✝³ : NormedAlgebra 𝕜₁ A
      inst✝² : NormedAlgebra 𝕜₂ A
      inst✝¹ : NormedAlgebra 𝕜₁ 𝕜₂
      inst✝ : IsScalarTower 𝕜₁ 𝕜₂ A
      f : 𝕜₂ → 𝕜₁
      a : A
      h : SpectrumRestricts a f
      this : ∀ (x : 𝕜₁), Eq (NNNorm.nnnorm ((algebraMap 𝕜₁ 𝕜₂) x)) (NNNorm.nnnorm x)
      x : 𝕜₂
      hx : Membership.mem (spectrum 𝕜₂ a) x
      ⊢ LE.le (↑(NNNorm.nnnorm x)) (iSup fun k => iSup fun h => ↑(NNNorm.nnnorm k))
    -/
  · have ⟨y, hy, hy'⟩ := h.algebraMap_image.symm ▸ hx
    /-
      𝕜₁ : Type u_3
      𝕜₂ : Type u_4
      A : Type u_5
      inst✝⁶ : NormedField 𝕜₁
      inst✝⁵ : NormedField 𝕜₂
      inst✝⁴ : NormedRing A
      inst✝³ : NormedAlgebra 𝕜₁ A
      inst✝² : NormedAlgebra 𝕜₂ A
      inst✝¹ : NormedAlgebra 𝕜₁ 𝕜₂
      inst✝ : IsScalarTower 𝕜₁ 𝕜₂ A
      f : 𝕜₂ → 𝕜₁
      a : A
      h : SpectrumRestricts a f
      this : ∀ (x : 𝕜₁), Eq (NNNorm.nnnorm ((algebraMap 𝕜₁ 𝕜₂) x)) (NNNorm.nnnorm x)
      x : 𝕜₂
      hx : Membership.mem (spectrum 𝕜₂ a) x
      y : 𝕜₁
      hy : Membership.mem (spectrum 𝕜₁ a) y
      hy' : Eq ((algebraMap 𝕜₁ 𝕜₂) y) x
      ⊢ LE.le (↑(NNNorm.nnnorm x)) (iSup fun k => iSup fun h => ↑(NNNorm.nnnorm k))
    -/
    subst hy'
    /-
      𝕜₁ : Type u_3
      𝕜₂ : Type u_4
      A : Type u_5
      inst✝⁶ : NormedField 𝕜₁
      inst✝⁵ : NormedField 𝕜₂
      inst✝⁴ : NormedRing A
      inst✝³ : NormedAlgebra 𝕜₁ A
      inst✝² : NormedAlgebra 𝕜₂ A
      inst✝¹ : NormedAlgebra 𝕜₁ 𝕜₂
      inst✝ : IsScalarTower 𝕜₁ 𝕜₂ A
      f : 𝕜₂ → 𝕜₁
      a : A
      h : SpectrumRestricts a f
      this : ∀ (x : 𝕜₁), Eq (NNNorm.nnnorm ((algebraMap 𝕜₁ 𝕜₂) x)) (NNNorm.nnnorm x)
      y : 𝕜₁
      hy : Membership.mem (spectrum 𝕜₁ a) y
      hx : Membership.mem (spectrum 𝕜₂ a) ((algebraMap 𝕜₁ 𝕜₂) y)
      ⊢ LE.le (↑(NNNorm.nnnorm ((algebraMap 𝕜₁ 𝕜₂) y))) (iSup fun k => iSup fun h => …
    -/
    exact this y ▸ le_iSup₂ (α := ℝ≥0∞) y hy
    /-
      🎉 no goals
    -/


lemma nnreal_iff [Algebra ℝ A] {a : A} :
    SpectrumRestricts a ContinuousMap.realToNNReal ↔ ∀ x ∈ spectrum ℝ a, 0 ≤ x := by
  /-
    A : Type u_3
    inst✝¹ : Ring A
    inst✝ : Algebra Real A
    a : A
    ⊢ Iff (SpectrumRestricts a ⇑ContinuousMap.realToNNReal) (∀ (x : Real), Members …
  -/
  refine ⟨fun h x hx ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      A : Type u_3
      inst✝¹ : Ring A
      inst✝ : Algebra Real A
      a : A
      h : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
      x : Real
      hx : Membership.mem (spectrum Real a) x
      ⊢ LE.le 0 x
    -/
  · obtain ⟨x, -, rfl⟩ := h.algebraMap_image.symm ▸ hx
    /-
      case refine_1.intro.intro
      A : Type u_3
      inst✝¹ : Ring A
      inst✝ : Algebra Real A
      a : A
      h : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
      x : NNReal
      hx : Membership.mem (spectrum Real a) ((algebraMap NNReal Real) x)
      ⊢ LE.le 0 ((algebraMap NNReal Real) x)
    -/
    exact coe_nonneg x
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      A : Type u_3
      inst✝¹ : Ring A
      inst✝ : Algebra Real A
      a : A
      h : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
      ⊢ SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    -/
  · exact .of_subset_range_algebraMap (fun _ ↦ Real.toNNReal_coe) fun x hx ↦ ⟨⟨x, h x hx⟩, rfl⟩
    /-
      🎉 no goals
    -/


lemma nnreal_of_nonneg {A : Type*} [Ring A] [PartialOrder A] [Algebra ℝ A]
    [NonnegSpectrumClass ℝ A] {a : A} (ha : 0 ≤ a) :
    SpectrumRestricts a ContinuousMap.realToNNReal :=
  nnreal_iff.mpr <| spectrum_nonneg_of_nonneg ha


lemma real_iff [Algebra ℂ A] {a : A} :
    SpectrumRestricts a Complex.reCLM ↔ ∀ x ∈ spectrum ℂ a, x = x.re := by
  /-
    A : Type u_3
    inst✝¹ : Ring A
    inst✝ : Algebra Complex A
    a : A
    ⊢ Iff (SpectrumRestricts a ⇑Complex.reCLM) (∀ (x : Complex), Membership.mem (s …
  -/
  refine ⟨fun h x hx ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      A : Type u_3
      inst✝¹ : Ring A
      inst✝ : Algebra Complex A
      a : A
      h : SpectrumRestricts a ⇑Complex.reCLM
      x : Complex
      hx : Membership.mem (spectrum Complex a) x
      ⊢ Eq x ↑x.re
    -/
  · obtain ⟨x, -, rfl⟩ := h.algebraMap_image.symm ▸ hx
    /-
      case refine_1.intro.intro
      A : Type u_3
      inst✝¹ : Ring A
      inst✝ : Algebra Complex A
      a : A
      h : SpectrumRestricts a ⇑Complex.reCLM
      x : Real
      hx : Membership.mem (spectrum Complex a) ((algebraMap Real Complex) x)
      ⊢ Eq ((algebraMap Real Complex) x) ↑((algebraMap Real Complex) x).re
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      A : Type u_3
      inst✝¹ : Ring A
      inst✝ : Algebra Complex A
      a : A
      h : ∀ (x : Complex), Membership.mem (spectrum Complex a) x → Eq x ↑x.re
      ⊢ SpectrumRestricts a ⇑Complex.reCLM
    -/
  · exact .of_subset_range_algebraMap Complex.ofReal_re fun x hx ↦ ⟨x.re, (h x hx).symm⟩
    /-
      🎉 no goals
    -/


lemma nnreal_le_iff [Algebra ℝ A] {a : A}
    (ha : SpectrumRestricts a ContinuousMap.realToNNReal) {r : ℝ≥0} :
    (∀ x ∈ spectrum ℝ≥0 a, r ≤ x) ↔ ∀ x ∈ spectrum ℝ a, r ≤ x := by
  /-
    A : Type u_3
    inst✝¹ : Ring A
    inst✝ : Algebra Real A
    a : A
    ha : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    r : NNReal
    ⊢ Iff (∀ (x : NNReal), Membership.mem (spectrum NNReal a) x → LE.le r x) (∀ (x …
  -/
  simp [← ha.algebraMap_image]
  /-
    🎉 no goals
  -/


lemma nnreal_lt_iff [Algebra ℝ A] {a : A}
    (ha : SpectrumRestricts a ContinuousMap.realToNNReal) {r : ℝ≥0} :
    (∀ x ∈ spectrum ℝ≥0 a, r < x) ↔ ∀ x ∈ spectrum ℝ a, r < x := by
  /-
    A : Type u_3
    inst✝¹ : Ring A
    inst✝ : Algebra Real A
    a : A
    ha : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    r : NNReal
    ⊢ Iff (∀ (x : NNReal), Membership.mem (spectrum NNReal a) x → LT.lt r x) (∀ (x …
  -/
  simp [← ha.algebraMap_image]
  /-
    🎉 no goals
  -/


lemma le_nnreal_iff [Algebra ℝ A] {a : A}
    (ha : SpectrumRestricts a ContinuousMap.realToNNReal) {r : ℝ≥0} :
    (∀ x ∈ spectrum ℝ≥0 a, x ≤ r) ↔ ∀ x ∈ spectrum ℝ a, x ≤ r := by
  /-
    A : Type u_3
    inst✝¹ : Ring A
    inst✝ : Algebra Real A
    a : A
    ha : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    r : NNReal
    ⊢ Iff (∀ (x : NNReal), Membership.mem (spectrum NNReal a) x → LE.le x r) (∀ (x …
  -/
  simp [← ha.algebraMap_image]
  /-
    🎉 no goals
  -/


lemma lt_nnreal_iff [Algebra ℝ A] {a : A}
    (ha : SpectrumRestricts a ContinuousMap.realToNNReal) {r : ℝ≥0} :
    (∀ x ∈ spectrum ℝ≥0 a, x < r) ↔ ∀ x ∈ spectrum ℝ a, x < r := by
  /-
    A : Type u_3
    inst✝¹ : Ring A
    inst✝ : Algebra Real A
    a : A
    ha : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    r : NNReal
    ⊢ Iff (∀ (x : NNReal), Membership.mem (spectrum NNReal a) x → LT.lt x r) (∀ (x …
  -/
  simp [← ha.algebraMap_image]
  /-
    🎉 no goals
  -/


lemma nnreal_iff_spectralRadius_le [Algebra ℝ A] {a : A} {t : ℝ≥0} (ht : spectralRadius ℝ a ≤ t) :
    SpectrumRestricts a ContinuousMap.realToNNReal ↔
      spectralRadius ℝ (algebraMap ℝ A t - a) ≤ t := by
  have : spectrum ℝ a ⊆ Set.Icc (-t) t := by
    intro x hx
    rw [Set.mem_Icc, ← abs_le, ← Real.norm_eq_abs, ← coe_nnnorm, NNReal.coe_le_coe,
      ← ENNReal.coe_le_coe]
    exact le_iSup₂ (α := ℝ≥0∞) x hx |>.trans ht
  /-
    A : Type u_3
    inst✝¹ : Ring A
    inst✝ : Algebra Real A
    a : A
    t : NNReal
    ht : LE.le (spectralRadius Real a) ↑t
    this : HasSubset.Subset (spectrum Real a) (Set.Icc (Neg.neg ↑t) ↑t)
    ⊢ Iff (SpectrumRestricts a ⇑ContinuousMap.realToNNReal) (LE.le (spectralRadius …
  -/
  rw [nnreal_iff]
  /-
    A : Type u_3
    inst✝¹ : Ring A
    inst✝ : Algebra Real A
    a : A
    t : NNReal
    ht : LE.le (spectralRadius Real a) ↑t
    this : HasSubset.Subset (spectrum Real a) (Set.Icc (Neg.neg ↑t) ↑t)
    ⊢ Iff (∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x) (LE.le (s …
  -/
  refine ⟨fun h ↦ iSup₂_le fun x hx ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      A : Type u_3
      inst✝¹ : Ring A
      inst✝ : Algebra Real A
      a : A
      t : NNReal
      ht : LE.le (spectralRadius Real a) ↑t
      this : HasSubset.Subset (spectrum Real a) (Set.Icc (Neg.neg ↑t) ↑t)
      h : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
      x : Real
      hx : Membership.mem (spectrum Real (HSub.hSub ((algebraMap Real A) ↑t) a)) x
      ⊢ LE.le ↑(NNNorm.nnnorm x) ↑t
    -/
  · rw [← spectrum.singleton_sub_eq] at hx
    /-
      case refine_1
      A : Type u_3
      inst✝¹ : Ring A
      inst✝ : Algebra Real A
      a : A
      t : NNReal
      ht : LE.le (spectralRadius Real a) ↑t
      this : HasSubset.Subset (spectrum Real a) (Set.Icc (Neg.neg ↑t) ↑t)
      h : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
      x : Real
      hx : Membership.mem (HSub.hSub (Singleton.singleton ↑t) (spectrum Real a)) x
      ⊢ LE.le ↑(NNNorm.nnnorm x) ↑t
    -/
    obtain ⟨y, hy, rfl⟩ : ∃ y ∈ spectrum ℝ a, ↑t - y = x := by simpa using hx
    /-
      case refine_1.intro.intro
      A : Type u_3
      inst✝¹ : Ring A
      inst✝ : Algebra Real A
      a : A
      t : NNReal
      ht : LE.le (spectralRadius Real a) ↑t
      this : HasSubset.Subset (spectrum Real a) (Set.Icc (Neg.neg ↑t) ↑t)
      h : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
      y : Real
      hy : Membership.mem (spectrum Real a) y
      hx : Membership.mem (HSub.hSub (Singleton.singleton ↑t) (spectrum Real a)) (HS …
      ⊢ LE.le ↑(NNNorm.nnnorm (HSub.hSub (↑t) y)) ↑t
    -/
    obtain ⟨hty, hyt⟩ := Set.mem_Icc.mp <| this hy
    /-
      case refine_1.intro.intro.intro
      A : Type u_3
      inst✝¹ : Ring A
      inst✝ : Algebra Real A
      a : A
      t : NNReal
      ht : LE.le (spectralRadius Real a) ↑t
      this : HasSubset.Subset (spectrum Real a) (Set.Icc (Neg.neg ↑t) ↑t)
      h : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
      y : Real
      hy : Membership.mem (spectrum Real a) y
      hx : Membership.mem (HSub.hSub (Singleton.singleton ↑t) (spectrum Real a)) (HS …
      hty : LE.le (Neg.neg ↑t) y
      hyt : LE.le y ↑t
      ⊢ LE.le ↑(NNNorm.nnnorm (HSub.hSub (↑t) y)) ↑t
    -/
    lift y to ℝ≥0 using h y hy
    /-
      case refine_1.intro.intro.intro.intro
      A : Type u_3
      inst✝¹ : Ring A
      inst✝ : Algebra Real A
      a : A
      t : NNReal
      ht : LE.le (spectralRadius Real a) ↑t
      this : HasSubset.Subset (spectrum Real a) (Set.Icc (Neg.neg ↑t) ↑t)
      h : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
      y : NNReal
      hy : Membership.mem (spectrum Real a) ↑y
      hx : Membership.mem (HSub.hSub (Singleton.singleton ↑t) (spectrum Real a)) (HS …
      hty : LE.le (Neg.neg ↑t) ↑y
      hyt : LE.le ↑y ↑t
      ⊢ LE.le ↑(NNNorm.nnnorm (HSub.hSub ↑t ↑y)) ↑t
    -/
    rw [← NNReal.coe_sub (by exact_mod_cast hyt)]
    /-
      case refine_1.intro.intro.intro.intro
      A : Type u_3
      inst✝¹ : Ring A
      inst✝ : Algebra Real A
      a : A
      t : NNReal
      ht : LE.le (spectralRadius Real a) ↑t
      this : HasSubset.Subset (spectrum Real a) (Set.Icc (Neg.neg ↑t) ↑t)
      h : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
      y : NNReal
      hy : Membership.mem (spectrum Real a) ↑y
      hx : Membership.mem (HSub.hSub (Singleton.singleton ↑t) (spectrum Real a)) (HS …
      hty : LE.le (Neg.neg ↑t) ↑y
      hyt : LE.le ↑y ↑t
      ⊢ LE.le ↑(NNNorm.nnnorm ↑(HSub.hSub t y)) ↑t
    -/
    simp
    /-
      🎉 no goals
    -/
  · replace h : ∀ x ∈ spectrum ℝ a, ‖t - x‖₊ ≤ t := by
      simpa [spectralRadius, iSup₂_le_iff, ← spectrum.singleton_sub_eq] using h
    /-
      case refine_2
      A : Type u_3
      inst✝¹ : Ring A
      inst✝ : Algebra Real A
      a : A
      t : NNReal
      ht : LE.le (spectralRadius Real a) ↑t
      this : HasSubset.Subset (spectrum Real a) (Set.Icc (Neg.neg ↑t) ↑t)
      h : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le (NNNorm.nnnorm (H …
      ⊢ ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le 0 x
    -/
    peel h with x hx h_le
    /-
      case refine_2.h.h
      A : Type u_3
      inst✝¹ : Ring A
      inst✝ : Algebra Real A
      a : A
      t : NNReal
      ht : LE.le (spectralRadius Real a) ↑t
      this : HasSubset.Subset (spectrum Real a) (Set.Icc (Neg.neg ↑t) ↑t)
      h : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le (NNNorm.nnnorm (H …
      x : Real
      hx : Membership.mem (spectrum Real a) x
      h_le : LE.le (NNNorm.nnnorm (HSub.hSub (↑t) x)) t
      ⊢ LE.le 0 x
    -/
    rw [← NNReal.coe_le_coe, coe_nnnorm, Real.norm_eq_abs, abs_le] at h_le
    /-
      case refine_2.h.h
      A : Type u_3
      inst✝¹ : Ring A
      inst✝ : Algebra Real A
      a : A
      t : NNReal
      ht : LE.le (spectralRadius Real a) ↑t
      this : HasSubset.Subset (spectrum Real a) (Set.Icc (Neg.neg ↑t) ↑t)
      h : ∀ (x : Real), Membership.mem (spectrum Real a) x → LE.le (NNNorm.nnnorm (H …
      x : Real
      hx : Membership.mem (spectrum Real a) x
      h_le : And (LE.le (Neg.neg ↑t) (HSub.hSub (↑t) x)) (LE.le (HSub.hSub (↑t) x) ↑t)
      ⊢ LE.le 0 x
    -/
    linarith [h_le.2]
    /-
      🎉 no goals
    -/


lemma _root_.NNReal.spectralRadius_mem_spectrum {A : Type*} [NormedRing A] [NormedAlgebra ℝ A]
    [CompleteSpace A] {a : A} (ha : (spectrum ℝ a).Nonempty)
    (ha' : SpectrumRestricts a ContinuousMap.realToNNReal) :
    (spectralRadius ℝ a).toNNReal ∈ spectrum ℝ≥0 a := by
  /-
    A : Type u_4
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    inst✝ : CompleteSpace A
    a : A
    ha : (spectrum Real a).Nonempty
    ha' : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    ⊢ Membership.mem (spectrum NNReal a) (spectralRadius Real a).toNNReal
  -/
  obtain ⟨x, hx₁, hx₂⟩ := spectrum.exists_nnnorm_eq_spectralRadius_of_nonempty ha
  /-
    case intro.intro
    A : Type u_4
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    inst✝ : CompleteSpace A
    a : A
    ha : (spectrum Real a).Nonempty
    ha' : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    x : Real
    hx₁ : Membership.mem (spectrum Real a) x
    hx₂ : Eq (↑(NNNorm.nnnorm x)) (spectralRadius Real a)
    ⊢ Membership.mem (spectrum NNReal a) (spectralRadius Real a).toNNReal
  -/
  rw [← hx₂, ENNReal.toNNReal_coe, ← spectrum.algebraMap_mem_iff ℝ, NNReal.algebraMap_eq_coe]
  /-
    case intro.intro
    A : Type u_4
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    inst✝ : CompleteSpace A
    a : A
    ha : (spectrum Real a).Nonempty
    ha' : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    x : Real
    hx₁ : Membership.mem (spectrum Real a) x
    hx₂ : Eq (↑(NNNorm.nnnorm x)) (spectralRadius Real a)
    ⊢ Membership.mem (spectrum Real a) ↑(NNNorm.nnnorm x)
  -/
  have : 0 ≤ x := ha'.rightInvOn hx₁ ▸ NNReal.zero_le_coe
  /-
    case intro.intro
    A : Type u_4
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    inst✝ : CompleteSpace A
    a : A
    ha : (spectrum Real a).Nonempty
    ha' : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    x : Real
    hx₁ : Membership.mem (spectrum Real a) x
    hx₂ : Eq (↑(NNNorm.nnnorm x)) (spectralRadius Real a)
    this : LE.le 0 x
    ⊢ Membership.mem (spectrum Real a) ↑(NNNorm.nnnorm x)
  -/
  convert hx₁
  /-
    case h.e'_5
    A : Type u_4
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    inst✝ : CompleteSpace A
    a : A
    ha : (spectrum Real a).Nonempty
    ha' : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    x : Real
    hx₁ : Membership.mem (spectrum Real a) x
    hx₂ : Eq (↑(NNNorm.nnnorm x)) (spectralRadius Real a)
    this : LE.le 0 x
    ⊢ Eq (↑(NNNorm.nnnorm x)) x
  -/
  simpa
  /-
    🎉 no goals
  -/


lemma _root_.Real.spectralRadius_mem_spectrum {A : Type*} [NormedRing A] [NormedAlgebra ℝ A]
    [CompleteSpace A] {a : A} (ha : (spectrum ℝ a).Nonempty)
    (ha' : SpectrumRestricts a ContinuousMap.realToNNReal) :
    (spectralRadius ℝ a).toReal ∈ spectrum ℝ a :=
  NNReal.spectralRadius_mem_spectrum ha ha'


lemma _root_.Real.spectralRadius_mem_spectrum_or {A : Type*} [NormedRing A] [NormedAlgebra ℝ A]
    [CompleteSpace A] {a : A} (ha : (spectrum ℝ a).Nonempty) :
    (spectralRadius ℝ a).toReal ∈ spectrum ℝ a ∨ -(spectralRadius ℝ a).toReal ∈ spectrum ℝ a := by
  /-
    A : Type u_4
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    inst✝ : CompleteSpace A
    a : A
    ha : (spectrum Real a).Nonempty
    ⊢ Or (Membership.mem (spectrum Real a) (spectralRadius Real a).toReal) (Member …
  -/
  obtain ⟨x, hx₁, hx₂⟩ := spectrum.exists_nnnorm_eq_spectralRadius_of_nonempty ha
  /-
    case intro.intro
    A : Type u_4
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    inst✝ : CompleteSpace A
    a : A
    ha : (spectrum Real a).Nonempty
    x : Real
    hx₁ : Membership.mem (spectrum Real a) x
    hx₂ : Eq (↑(NNNorm.nnnorm x)) (spectralRadius Real a)
    ⊢ Or (Membership.mem (spectrum Real a) (spectralRadius Real a).toReal) (Member …
  -/
  simp only [← hx₂, ENNReal.coe_toReal, coe_nnnorm, Real.norm_eq_abs]
  /-
    case intro.intro
    A : Type u_4
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra Real A
    inst✝ : CompleteSpace A
    a : A
    ha : (spectrum Real a).Nonempty
    x : Real
    hx₁ : Membership.mem (spectrum Real a) x
    hx₂ : Eq (↑(NNNorm.nnnorm x)) (spectralRadius Real a)
    ⊢ Or (Membership.mem (spectrum Real a) (abs x)) (Membership.mem (spectrum Real …
  -/
  exact abs_choice x |>.imp (fun h ↦ by rwa [h]) (fun h ↦ by simpa [h])
  /-
    🎉 no goals
  -/


local notation "σₙ" => quasispectrum


lemma compactSpace {R S A : Type*} [Semifield R] [Field S] [NonUnitalRing A]
    [Algebra R S] [Module R A] [Module S A] [IsScalarTower S A A] [SMulCommClass S A A]
    [IsScalarTower R S A] [TopologicalSpace R] [TopologicalSpace S] {a : A} (f : C(S, R))
    (h : QuasispectrumRestricts a f) [h_cpct : CompactSpace (σₙ S a)] :
    CompactSpace (σₙ R a) := by
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝¹⁰ : Semifield R
    inst✝⁹ : Field S
    inst✝⁸ : NonUnitalRing A
    inst✝⁷ : Algebra R S
    inst✝⁶ : Module R A
    inst✝⁵ : Module S A
    inst✝⁴ : IsScalarTower S A A
    inst✝³ : SMulCommClass S A A
    inst✝² : IsScalarTower R S A
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSpace S
    a : A
    f : ContinuousMap S R
    h : QuasispectrumRestricts a ⇑f
    h_cpct : CompactSpace ↑(quasispectrum S a)
    ⊢ CompactSpace ↑(quasispectrum R a)
  -/
  rw [← isCompact_iff_compactSpace] at h_cpct ⊢
  /-
    R : Type u_3
    S : Type u_4
    A : Type u_5
    inst✝¹⁰ : Semifield R
    inst✝⁹ : Field S
    inst✝⁸ : NonUnitalRing A
    inst✝⁷ : Algebra R S
    inst✝⁶ : Module R A
    inst✝⁵ : Module S A
    inst✝⁴ : IsScalarTower S A A
    inst✝³ : SMulCommClass S A A
    inst✝² : IsScalarTower R S A
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSpace S
    a : A
    f : ContinuousMap S R
    h : QuasispectrumRestricts a ⇑f
    h_cpct : IsCompact (quasispectrum S a)
    ⊢ IsCompact (quasispectrum R a)
  -/
  exact h.image ▸ h_cpct.image (map_continuous f)
  /-
    🎉 no goals
  -/


lemma nnreal_iff [Module ℝ A] [IsScalarTower ℝ A A] [SMulCommClass ℝ A A] {a : A} :
    QuasispectrumRestricts a ContinuousMap.realToNNReal ↔ ∀ x ∈ σₙ ℝ a, 0 ≤ x := by
  rw [quasispectrumRestricts_iff_spectrumRestricts_inr,
    Unitization.quasispectrum_eq_spectrum_inr' _ ℝ, SpectrumRestricts.nnreal_iff]


lemma nnreal_of_nonneg [Module ℝ A] [IsScalarTower ℝ A A] [SMulCommClass ℝ A A] [PartialOrder A]
    [NonnegSpectrumClass ℝ A] {a : A} (ha : 0 ≤ a) :
    QuasispectrumRestricts a ContinuousMap.realToNNReal :=
  nnreal_iff.mpr <| quasispectrum_nonneg_of_nonneg _ ha


lemma real_iff [Module ℂ A] [IsScalarTower ℂ A A] [SMulCommClass ℂ A A] {a : A} :
    QuasispectrumRestricts a Complex.reCLM ↔ ∀ x ∈ σₙ ℂ a, x = x.re := by
  rw [quasispectrumRestricts_iff_spectrumRestricts_inr,
    Unitization.quasispectrum_eq_spectrum_inr' _ ℂ, SpectrumRestricts.real_iff]


lemma le_nnreal_iff [Module ℝ A] [IsScalarTower ℝ A A] [SMulCommClass ℝ A A] {a : A}
    (ha : QuasispectrumRestricts a ContinuousMap.realToNNReal) {r : ℝ≥0} :
    (∀ x ∈ quasispectrum ℝ≥0 a, x ≤ r) ↔ ∀ x ∈ quasispectrum ℝ a, x ≤ r := by
  /-
    A : Type u_3
    inst✝³ : NonUnitalRing A
    inst✝² : Module Real A
    inst✝¹ : IsScalarTower Real A A
    inst✝ : SMulCommClass Real A A
    a : A
    ha : QuasispectrumRestricts a ⇑ContinuousMap.realToNNReal
    r : NNReal
    ⊢ Iff (∀ (x : NNReal), Membership.mem (quasispectrum NNReal a) x → LE.le x r)  …
  -/
  simp [← ha.algebraMap_image]
  /-
    🎉 no goals
  -/


lemma lt_nnreal_iff [Module ℝ A] [IsScalarTower ℝ A A] [SMulCommClass ℝ A A] {a : A}
    (ha : QuasispectrumRestricts a ContinuousMap.realToNNReal) {r : ℝ≥0} :
    (∀ x ∈ quasispectrum ℝ≥0 a, x < r) ↔ ∀ x ∈ quasispectrum ℝ a, x < r := by
  /-
    A : Type u_3
    inst✝³ : NonUnitalRing A
    inst✝² : Module Real A
    inst✝¹ : IsScalarTower Real A A
    inst✝ : SMulCommClass Real A A
    a : A
    ha : QuasispectrumRestricts a ⇑ContinuousMap.realToNNReal
    r : NNReal
    ⊢ Iff (∀ (x : NNReal), Membership.mem (quasispectrum NNReal a) x → LT.lt x r)  …
  -/
  simp [← ha.algebraMap_image]
  /-
    🎉 no goals
  -/


lemma coe_mem_spectrum_real_of_nonneg [Algebra ℝ A] [NonnegSpectrumClass ℝ A] {a : A} {x : ℝ≥0}
    (ha : 0 ≤ a := by cfc_tac) :
    (x : ℝ) ∈ spectrum ℝ a ↔ x ∈ spectrum ℝ≥0 a := by
  simp [← (SpectrumRestricts.nnreal_of_nonneg ha).algebraMap_image, Set.mem_image,
    NNReal.algebraMap_eq_coe]


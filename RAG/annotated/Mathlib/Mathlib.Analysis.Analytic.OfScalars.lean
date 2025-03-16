/-- Formal power series of `∑ cᵢ • xⁱ` for some scalar field `𝕜` and ring algebra `E`-/
def ofScalars (c : ℕ → 𝕜) : FormalMultilinearSeries 𝕜 E E :=
  fun n ↦ c n • ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n E


@[simp]
theorem ofScalars_eq_zero [Nontrivial E] (n : ℕ) : ofScalars E c n = 0 ↔ c n = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : Field 𝕜
    inst✝⁴ : Ring E
    inst✝³ : Algebra 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalRing E
    c : Nat → 𝕜
    inst✝ : Nontrivial E
    n : Nat
    ⊢ Iff (Eq (FormalMultilinearSeries.ofScalars E c n) 0) (Eq (c n) 0)
  -/
  rw [ofScalars, smul_eq_zero (c := c n) (x := ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n E)]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : Field 𝕜
    inst✝⁴ : Ring E
    inst✝³ : Algebra 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalRing E
    c : Nat → 𝕜
    inst✝ : Nontrivial E
    n : Nat
    ⊢ Iff (Or (Eq (c n) 0) (Eq (ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n E) 0)) …
  -/
  refine or_iff_left (ContinuousMultilinearMap.ext_iff.1.mt <| not_forall_of_exists_not ?_)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : Field 𝕜
    inst✝⁴ : Ring E
    inst✝³ : Algebra 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalRing E
    c : Nat → 𝕜
    inst✝ : Nontrivial E
    n : Nat
    ⊢ Exists fun x => Not (Eq ((ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n E) x)  …
  -/
  use fun _ ↦ 1
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : Field 𝕜
    inst✝⁴ : Ring E
    inst✝³ : Algebra 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalRing E
    c : Nat → 𝕜
    inst✝ : Nontrivial E
    n : Nat
    ⊢ Not (Eq ((ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n E) fun x => 1) (0 fun  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem ofScalars_eq_zero_of_scalar_zero {n : ℕ} (hc : c n = 0) : ofScalars E c n = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : Field 𝕜
    inst✝³ : Ring E
    inst✝² : Algebra 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalRing E
    c : Nat → 𝕜
    n : Nat
    hc : Eq (c n) 0
    ⊢ Eq (FormalMultilinearSeries.ofScalars E c n) 0
  -/
  rw [ofScalars, hc, zero_smul 𝕜 (ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n E)]
  /-
    🎉 no goals
  -/


@[simp]
theorem ofScalars_series_eq_zero [Nontrivial E] : ofScalars E c = 0 ↔ c = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : Field 𝕜
    inst✝⁴ : Ring E
    inst✝³ : Algebra 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalRing E
    c : Nat → 𝕜
    inst✝ : Nontrivial E
    ⊢ Iff (Eq (FormalMultilinearSeries.ofScalars E c) 0) (Eq c 0)
  -/
  simp [FormalMultilinearSeries.ext_iff, funext_iff]
  /-
    🎉 no goals
  -/


variable (𝕜) in
@[simp]
theorem ofScalars_series_eq_zero_of_scalar_zero : ofScalars E (0 : ℕ → 𝕜) = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : Field 𝕜
    inst✝³ : Ring E
    inst✝² : Algebra 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalRing E
    ⊢ Eq (FormalMultilinearSeries.ofScalars E 0) 0
  -/
  simp [FormalMultilinearSeries.ext_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem ofScalars_series_of_subsingleton [Subsingleton E] : ofScalars E c = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : Field 𝕜
    inst✝⁴ : Ring E
    inst✝³ : Algebra 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalRing E
    c : Nat → 𝕜
    inst✝ : Subsingleton E
    ⊢ Eq (FormalMultilinearSeries.ofScalars E c) 0
  -/
  simp_rw [FormalMultilinearSeries.ext_iff, ofScalars, ContinuousMultilinearMap.ext_iff]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : Field 𝕜
    inst✝⁴ : Ring E
    inst✝³ : Algebra 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalRing E
    c : Nat → 𝕜
    inst✝ : Subsingleton E
    ⊢ ∀ (n : Nat) (x : Fin n → E), Eq ((HSMul.hSMul (c n) (ContinuousMultilinearMa …
  -/
  exact fun _ _ ↦ Subsingleton.allEq _ _
  /-
    🎉 no goals
  -/


variable (𝕜) in
theorem ofScalars_series_injective [Nontrivial E] : Function.Injective (ofScalars E (𝕜 := 𝕜)) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : Field 𝕜
    inst✝⁴ : Ring E
    inst✝³ : Algebra 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalRing E
    inst✝ : Nontrivial E
    ⊢ Function.Injective (FormalMultilinearSeries.ofScalars E)
  -/
  intro _ _
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : Field 𝕜
    inst✝⁴ : Ring E
    inst✝³ : Algebra 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalRing E
    inst✝ : Nontrivial E
    a₁✝ a₂✝ : Nat → 𝕜
    ⊢ Eq (FormalMultilinearSeries.ofScalars E a₁✝) (FormalMultilinearSeries.ofScal …
  -/
  refine Function.mtr fun h ↦ ?_
  simp_rw [FormalMultilinearSeries.ext_iff, ofScalars, ContinuousMultilinearMap.ext_iff,
    ContinuousMultilinearMap.smul_apply]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : Field 𝕜
    inst✝⁴ : Ring E
    inst✝³ : Algebra 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalRing E
    inst✝ : Nontrivial E
    a₁✝ a₂✝ : Nat → 𝕜
    h : Not (Eq a₁✝ a₂✝)
    ⊢ Not (∀ (n : Nat) (x : Fin n → E), Eq (HSMul.hSMul (a₁✝ n) ((ContinuousMultil …
  -/
  push_neg
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : Field 𝕜
    inst✝⁴ : Ring E
    inst✝³ : Algebra 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalRing E
    inst✝ : Nontrivial E
    a₁✝ a₂✝ : Nat → 𝕜
    h : Not (Eq a₁✝ a₂✝)
    ⊢ Exists fun n => Exists fun x => Ne (HSMul.hSMul (a₁✝ n) ((ContinuousMultilin …
  -/
  obtain ⟨n, hn⟩ := Function.ne_iff.1 h
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : Field 𝕜
    inst✝⁴ : Ring E
    inst✝³ : Algebra 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalRing E
    inst✝ : Nontrivial E
    a₁✝ a₂✝ : Nat → 𝕜
    h : Not (Eq a₁✝ a₂✝)
    n : Nat
    hn : Ne (a₁✝ n) (a₂✝ n)
    ⊢ Exists fun n => Exists fun x => Ne (HSMul.hSMul (a₁✝ n) ((ContinuousMultilin …
  -/
  refine ⟨n, fun _ ↦ 1, ?_⟩
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : Field 𝕜
    inst✝⁴ : Ring E
    inst✝³ : Algebra 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalRing E
    inst✝ : Nontrivial E
    a₁✝ a₂✝ : Nat → 𝕜
    h : Not (Eq a₁✝ a₂✝)
    n : Nat
    hn : Ne (a₁✝ n) (a₂✝ n)
    ⊢ Ne (HSMul.hSMul (a₁✝ n) ((ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n E) fun …
  -/
  simp only [mkPiAlgebraFin_apply, List.ofFn_const, List.prod_replicate, one_pow, ne_eq]
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : Field 𝕜
    inst✝⁴ : Ring E
    inst✝³ : Algebra 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalRing E
    inst✝ : Nontrivial E
    a₁✝ a₂✝ : Nat → 𝕜
    h : Not (Eq a₁✝ a₂✝)
    n : Nat
    hn : Ne (a₁✝ n) (a₂✝ n)
    ⊢ Not (Eq (HSMul.hSMul (a₁✝ n) 1) (HSMul.hSMul (a₂✝ n) 1))
  -/
  exact (smul_left_injective 𝕜 one_ne_zero).ne hn
  /-
    🎉 no goals
  -/


@[simp]
theorem ofScalars_series_eq_iff [Nontrivial E] (c' : ℕ → 𝕜) :
    ofScalars E c = ofScalars E c' ↔ c = c' :=
  ⟨fun e => ofScalars_series_injective 𝕜 E e, _root_.congrArg _⟩


theorem ofScalars_apply_zero (n : ℕ) :
    (ofScalars E c n fun _ => 0) = Pi.single (f := fun _ => E) 0 (c 0 • 1) n := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : Field 𝕜
    inst✝³ : Ring E
    inst✝² : Algebra 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalRing E
    c : Nat → 𝕜
    n : Nat
    ⊢ Eq ((FormalMultilinearSeries.ofScalars E c n) fun x => 0) (Pi.single 0 (HSMu …
  -/
  rw [ofScalars]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : Field 𝕜
    inst✝³ : Ring E
    inst✝² : Algebra 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalRing E
    c : Nat → 𝕜
    n : Nat
    ⊢ Eq ((HSMul.hSMul (c n) (ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n E)) fun  …
  -/
              /-
                🎉 no goals
              -/
  cases n <;> simp
              /-
                🎉 no goals
              -/


theorem ofScalars_add (c' : ℕ → 𝕜) : ofScalars E (c + c') = ofScalars E c + ofScalars E c' := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : Field 𝕜
    inst✝³ : Ring E
    inst✝² : Algebra 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalRing E
    c c' : Nat → 𝕜
    ⊢ Eq (FormalMultilinearSeries.ofScalars E (HAdd.hAdd c c')) (HAdd.hAdd (Formal …
  -/
  unfold ofScalars
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : Field 𝕜
    inst✝³ : Ring E
    inst✝² : Algebra 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalRing E
    c c' : Nat → 𝕜
    ⊢ Eq (fun n => HSMul.hSMul (HAdd.hAdd c c' n) (ContinuousMultilinearMap.mkPiAl …
  -/
  simp_rw [Pi.add_apply, Pi.add_def _ _]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : Field 𝕜
    inst✝³ : Ring E
    inst✝² : Algebra 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalRing E
    c c' : Nat → 𝕜
    ⊢ Eq (fun n => HSMul.hSMul (HAdd.hAdd (c n) (c' n)) (ContinuousMultilinearMap. …
  -/
  exact funext fun n ↦ Module.add_smul (c n) (c' n) (ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n E)
  /-
    🎉 no goals
  -/


theorem ofScalars_smul (x : 𝕜) : ofScalars E (x • c) = x • ofScalars E c := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : Field 𝕜
    inst✝³ : Ring E
    inst✝² : Algebra 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalRing E
    c : Nat → 𝕜
    x : 𝕜
    ⊢ Eq (FormalMultilinearSeries.ofScalars E (HSMul.hSMul x c)) (HSMul.hSMul x (F …
  -/
  unfold ofScalars
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : Field 𝕜
    inst✝³ : Ring E
    inst✝² : Algebra 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalRing E
    c : Nat → 𝕜
    x : 𝕜
    ⊢ Eq (fun n => HSMul.hSMul (HSMul.hSMul x c n) (ContinuousMultilinearMap.mkPiA …
  -/
  simp [Pi.smul_def x _, smul_smul]
  /-
    🎉 no goals
  -/


variable (𝕜) in
/-- The submodule generated by scalar series on `FormalMultilinearSeries 𝕜 E E`. -/
def ofScalarsSubmodule : Submodule 𝕜 (FormalMultilinearSeries 𝕜 E E) where
  carrier := {ofScalars E f | f}
  add_mem' := fun ⟨c, hc⟩ ⟨c', hc'⟩ ↦ ⟨c + c', hc' ▸ hc ▸ ofScalars_add E c c'⟩
  zero_mem' := ⟨0, ofScalars_series_eq_zero_of_scalar_zero 𝕜 E⟩
  smul_mem' := fun x _ ⟨c, hc⟩ ↦ ⟨x • c, hc ▸ ofScalars_smul E c x⟩


theorem ofScalars_apply_eq (x : E) (n : ℕ) :
    ofScalars E c n (fun _ ↦ x) = c n • x ^ n := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : Field 𝕜
    inst✝³ : Ring E
    inst✝² : Algebra 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalRing E
    c : Nat → 𝕜
    x : E
    n : Nat
    ⊢ Eq ((FormalMultilinearSeries.ofScalars E c n) fun x_1 => x) (HSMul.hSMul (c  …
  -/
  simp [ofScalars]
  /-
    🎉 no goals
  -/


/-- This naming follows the convention of `NormedSpace.expSeries_apply_eq'`. -/
theorem ofScalars_apply_eq' (x : E) :
    (fun n ↦ ofScalars E c n (fun _ ↦ x)) = fun n ↦ c n • x ^ n := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : Field 𝕜
    inst✝³ : Ring E
    inst✝² : Algebra 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalRing E
    c : Nat → 𝕜
    x : E
    ⊢ Eq (fun n => (FormalMultilinearSeries.ofScalars E c n) fun x_1 => x) fun n = …
  -/
  simp [ofScalars]
  /-
    🎉 no goals
  -/


/-- The sum of the formal power series. Takes the value `0` outside the radius of convergence. -/
noncomputable def ofScalarsSum := (ofScalars E c).sum


theorem ofScalars_sum_eq (x : E) : ofScalarsSum c x =
    ∑' n, c n • x ^ n := tsum_congr fun n => ofScalars_apply_eq c x n


theorem ofScalarsSum_eq_tsum : ofScalarsSum c =
    fun (x : E) => ∑' n : ℕ, c n • x ^ n := funext (ofScalars_sum_eq c)


@[simp]
theorem ofScalarsSum_zero : ofScalarsSum c (0 : E) = c 0 • 1 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : Field 𝕜
    inst✝³ : Ring E
    inst✝² : Algebra 𝕜 E
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalRing E
    c : Nat → 𝕜
    ⊢ Eq (FormalMultilinearSeries.ofScalarsSum c 0) (HSMul.hSMul (c 0) 1)
  -/
  simp [ofScalarsSum_eq_tsum, ← ofScalars_apply_eq, ofScalars_apply_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem ofScalarsSum_of_subsingleton [Subsingleton E] {x : E} : ofScalarsSum c x = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : Field 𝕜
    inst✝⁴ : Ring E
    inst✝³ : Algebra 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalRing E
    c : Nat → 𝕜
    inst✝ : Subsingleton E
    x : E
    ⊢ Eq (FormalMultilinearSeries.ofScalarsSum c x) 0
  -/
  simp [Subsingleton.eq_zero x, Subsingleton.eq_zero (1 : E)]
  /-
    🎉 no goals
  -/


@[simp]
theorem ofScalarsSum_op [T2Space E] (x : E) :
    ofScalarsSum c (MulOpposite.op x) = MulOpposite.op (ofScalarsSum c x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : Field 𝕜
    inst✝⁴ : Ring E
    inst✝³ : Algebra 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalRing E
    c : Nat → 𝕜
    inst✝ : T2Space E
    x : E
    ⊢ Eq (FormalMultilinearSeries.ofScalarsSum c (MulOpposite.op x)) (MulOpposite. …
  -/
  simp [ofScalars, ofScalars_sum_eq, ← MulOpposite.op_pow, ← MulOpposite.op_smul, tsum_op]
  /-
    🎉 no goals
  -/


@[simp]
theorem ofScalarsSum_unop [T2Space E] (x : Eᵐᵒᵖ) :
    ofScalarsSum c (MulOpposite.unop x) = MulOpposite.unop (ofScalarsSum c x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁵ : Field 𝕜
    inst✝⁴ : Ring E
    inst✝³ : Algebra 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalRing E
    c : Nat → 𝕜
    inst✝ : T2Space E
    x : MulOpposite E
    ⊢ Eq (FormalMultilinearSeries.ofScalarsSum c (MulOpposite.unop x)) (MulOpposit …
  -/
  simp [ofScalars, ofScalars_sum_eq, ← MulOpposite.unop_pow, ← MulOpposite.unop_smul, tsum_unop]
  /-
    🎉 no goals
  -/


set_option maxSynthPendingDepth 2 in
theorem ofScalars_norm_eq_mul :
    ‖ofScalars E c n‖ = ‖c n‖ * ‖ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n E‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedRing E
    inst✝ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    n : Nat
    ⊢ Eq (Norm.norm (FormalMultilinearSeries.ofScalars E c n)) (HMul.hMul (Norm.no …
  -/
  rw [ofScalars, norm_smul (c n) (ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n E)]
  /-
    🎉 no goals
  -/


theorem ofScalars_norm_le (hn : n > 0) : ‖ofScalars E c n‖ ≤ ‖c n‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedRing E
    inst✝ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    n : Nat
    hn : GT.gt n 0
    ⊢ LE.le (Norm.norm (FormalMultilinearSeries.ofScalars E c n)) (Norm.norm (c n))
  -/
  simp only [ofScalars_norm_eq_mul]
  exact (mul_le_of_le_one_right (norm_nonneg _)
    (ContinuousMultilinearMap.norm_mkPiAlgebraFin_le_of_pos hn))


@[simp]
theorem ofScalars_norm [NormOneClass E] : ‖ofScalars E c n‖ = ‖c n‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    n : Nat
    inst✝ : NormOneClass E
    ⊢ Eq (Norm.norm (FormalMultilinearSeries.ofScalars E c n)) (Norm.norm (c n))
  -/
  simp [ofScalars_norm_eq_mul]
  /-
    🎉 no goals
  -/


private theorem tendsto_succ_norm_div_norm {r r' : ℝ≥0} (hr' : r' ≠ 0)
    (hc : Tendsto (fun n ↦ ‖c n.succ‖ / ‖c n‖) atTop (𝓝 r)) :
      Tendsto (fun n ↦ ‖‖c (n + 1)‖ * r' ^ (n + 1)‖ /
        ‖‖c n‖ * r' ^ n‖) atTop (𝓝 ↑(r' * r)) := by
  simp_rw [norm_mul, norm_norm, mul_div_mul_comm, ← norm_div, pow_succ, mul_div_right_comm,
    div_self (pow_ne_zero _ (NNReal.coe_ne_zero.mpr hr')), one_mul, norm_div, NNReal.norm_eq]
  /-
    𝕜 : Type u_1
    inst✝ : NontriviallyNormedField 𝕜
    c : Nat → 𝕜
    r r' : NNReal
    hr' : Ne r' 0
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    ⊢ Filter.Tendsto (fun n => HMul.hMul (HDiv.hDiv (Norm.norm (c (HAdd.hAdd n 1)) …
  -/
  exact mul_comm r' r ▸ hc.mul tendsto_const_nhds
  /-
    🎉 no goals
  -/


theorem ofScalars_radius_ge_inv_of_tendsto {r : ℝ≥0} (hr : r ≠ 0)
    (hc : Tendsto (fun n ↦ ‖c n.succ‖ / ‖c n‖) atTop (𝓝 r)) :
      (ofScalars E c).radius ≥ ofNNReal r⁻¹ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedRing E
    inst✝ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    r : NNReal
    hr : Ne r 0
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    ⊢ GE.ge (FormalMultilinearSeries.ofScalars E c).radius ↑(Inv.inv r)
  -/
  refine le_of_forall_nnreal_lt (fun r' hr' ↦ ?_)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedRing E
    inst✝ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    r : NNReal
    hr : Ne r 0
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    r' : NNReal
    hr' : LT.lt ↑r' ↑(Inv.inv r)
    ⊢ LE.le (↑r') (FormalMultilinearSeries.ofScalars E c).radius
  -/
  rw [coe_lt_coe, NNReal.lt_inv_iff_mul_lt hr] at hr'
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedRing E
    inst✝ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    r : NNReal
    hr : Ne r 0
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    r' : NNReal
    hr' : LT.lt (HMul.hMul r' r) 1
    ⊢ LE.le (↑r') (FormalMultilinearSeries.ofScalars E c).radius
  -/
  by_cases hrz : r' = 0
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedRing E
      inst✝ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      r : NNReal
      hr : Ne r 0
      hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
      r' : NNReal
      hr' : LT.lt (HMul.hMul r' r) 1
      hrz : Eq r' 0
      ⊢ LE.le (↑r') (FormalMultilinearSeries.ofScalars E c).radius
    -/
  · simp [hrz]
    /-
      🎉 no goals
    -/
  /-
    case neg
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedRing E
    inst✝ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    r : NNReal
    hr : Ne r 0
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    r' : NNReal
    hr' : LT.lt (HMul.hMul r' r) 1
    hrz : Not (Eq r' 0)
    ⊢ LE.le (↑r') (FormalMultilinearSeries.ofScalars E c).radius
  -/
  apply FormalMultilinearSeries.le_radius_of_summable_norm
  /-
    case neg.hs
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedRing E
    inst✝ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    r : NNReal
    hr : Ne r 0
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    r' : NNReal
    hr' : LT.lt (HMul.hMul r' r) 1
    hrz : Not (Eq r' 0)
    ⊢ Summable fun n => HMul.hMul (Norm.norm (FormalMultilinearSeries.ofScalars E  …
  -/
  refine Summable.of_norm_bounded_eventually (fun n ↦ ‖‖c n‖ * r' ^ n‖) ?_ ?_
    /-
      case neg.hs.refine_1
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedRing E
      inst✝ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      r : NNReal
      hr : Ne r 0
      hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
      r' : NNReal
      hr' : LT.lt (HMul.hMul r' r) 1
      hrz : Not (Eq r' 0)
      ⊢ Summable fun n => Norm.norm (HMul.hMul (Norm.norm (c n)) (HPow.hPow (↑r') n))
    -/
  · refine summable_of_ratio_test_tendsto_lt_one hr' ?_ ?_
      /-
        case neg.hs.refine_1.refine_1
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedRing E
        inst✝ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        r : NNReal
        hr : Ne r 0
        hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
        r' : NNReal
        hr' : LT.lt (HMul.hMul r' r) 1
        hrz : Not (Eq r' 0)
        ⊢ Filter.Eventually (fun n => Ne (Norm.norm (HMul.hMul (Norm.norm (c n)) (HPow …
      -/
    · refine (hc.eventually_ne (NNReal.coe_ne_zero.mpr hr)).mp (Eventually.of_forall ?_)
      /-
        case neg.hs.refine_1.refine_1
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedRing E
        inst✝ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        r : NNReal
        hr : Ne r 0
        hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
        r' : NNReal
        hr' : LT.lt (HMul.hMul r' r) 1
        hrz : Not (Eq r' 0)
        ⊢ ∀ (x : Nat), Ne (HDiv.hDiv (Norm.norm (c x.succ)) (Norm.norm (c x))) 0 → Ne  …
      -/
      aesop
      /-
        🎉 no goals
      -/
      /-
        case neg.hs.refine_1.refine_2
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedRing E
        inst✝ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        r : NNReal
        hr : Ne r 0
        hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
        r' : NNReal
        hr' : LT.lt (HMul.hMul r' r) 1
        hrz : Not (Eq r' 0)
        ⊢ Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (Norm.norm (HMul.hMul (Norm.no …
      -/
    · simp_rw [norm_norm]
      /-
        case neg.hs.refine_1.refine_2
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedRing E
        inst✝ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        r : NNReal
        hr : Ne r 0
        hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
        r' : NNReal
        hr' : LT.lt (HMul.hMul r' r) 1
        hrz : Not (Eq r' 0)
        ⊢ Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (HMul.hMul (Norm.norm (c (HAdd …
      -/
      exact tendsto_succ_norm_div_norm c hrz hc
      /-
        🎉 no goals
      -/
    /-
      case neg.hs.refine_2
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedRing E
      inst✝ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      r : NNReal
      hr : Ne r 0
      hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
      r' : NNReal
      hr' : LT.lt (HMul.hMul r' r) 1
      hrz : Not (Eq r' 0)
      ⊢ Filter.Eventually (fun i => LE.le (Norm.norm (HMul.hMul (Norm.norm (FormalMu …
    -/
  · filter_upwards [eventually_cofinite_ne 0] with n hn
    /-
      case h
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedRing E
      inst✝ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      r : NNReal
      hr : Ne r 0
      hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
      r' : NNReal
      hr' : LT.lt (HMul.hMul r' r) 1
      hrz : Not (Eq r' 0)
      n : Nat
      hn : Ne n 0
      ⊢ LE.le (Norm.norm (HMul.hMul (Norm.norm (FormalMultilinearSeries.ofScalars E  …
    -/
    simp only [norm_mul, norm_norm, norm_pow, NNReal.norm_eq]
    /-
      case h
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedRing E
      inst✝ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      r : NNReal
      hr : Ne r 0
      hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
      r' : NNReal
      hr' : LT.lt (HMul.hMul r' r) 1
      hrz : Not (Eq r' 0)
      n : Nat
      hn : Ne n 0
      ⊢ LE.le (HMul.hMul (Norm.norm (FormalMultilinearSeries.ofScalars E c n)) (HPow …
    -/
    gcongr
    /-
      case h.h
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedRing E
      inst✝ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      r : NNReal
      hr : Ne r 0
      hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
      r' : NNReal
      hr' : LT.lt (HMul.hMul r' r) 1
      hrz : Not (Eq r' 0)
      n : Nat
      hn : Ne n 0
      ⊢ LE.le (Norm.norm (FormalMultilinearSeries.ofScalars E c n)) (Norm.norm (c n))
    -/
    exact ofScalars_norm_le E c n (Nat.pos_iff_ne_zero.mpr hn)
    /-
      🎉 no goals
    -/


/-- The radius of convergence of a scalar series is the inverse of the non-zero limit
`fun n ↦ ‖c n.succ‖ / ‖c n‖`. -/
theorem ofScalars_radius_eq_inv_of_tendsto [NormOneClass E] {r : ℝ≥0} (hr : r ≠ 0)
    (hc : Tendsto (fun n ↦ ‖c n.succ‖ / ‖c n‖) atTop (𝓝 r)) :
      (ofScalars E c).radius = ofNNReal r⁻¹ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    inst✝ : NormOneClass E
    r : NNReal
    hr : Ne r 0
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    ⊢ Eq (FormalMultilinearSeries.ofScalars E c).radius ↑(Inv.inv r)
  -/
  refine le_antisymm ?_ (ofScalars_radius_ge_inv_of_tendsto E c hr hc)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    inst✝ : NormOneClass E
    r : NNReal
    hr : Ne r 0
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    ⊢ LE.le (FormalMultilinearSeries.ofScalars E c).radius ↑(Inv.inv r)
  -/
  refine le_of_forall_nnreal_lt (fun r' hr' ↦ ?_)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    inst✝ : NormOneClass E
    r : NNReal
    hr : Ne r 0
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    r' : NNReal
    hr' : LT.lt (↑r') (FormalMultilinearSeries.ofScalars E c).radius
    ⊢ LE.le ↑r' ↑(Inv.inv r)
  -/
  rw [coe_le_coe, NNReal.le_inv_iff_mul_le hr]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    inst✝ : NormOneClass E
    r : NNReal
    hr : Ne r 0
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    r' : NNReal
    hr' : LT.lt (↑r') (FormalMultilinearSeries.ofScalars E c).radius
    ⊢ LE.le (HMul.hMul r' r) 1
  -/
  have := FormalMultilinearSeries.summable_norm_mul_pow _ hr'
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    inst✝ : NormOneClass E
    r : NNReal
    hr : Ne r 0
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    r' : NNReal
    hr' : LT.lt (↑r') (FormalMultilinearSeries.ofScalars E c).radius
    this : Summable fun n => HMul.hMul (Norm.norm (FormalMultilinearSeries.ofScala …
    ⊢ LE.le (HMul.hMul r' r) 1
  -/
  contrapose! this
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    inst✝ : NormOneClass E
    r : NNReal
    hr : Ne r 0
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    r' : NNReal
    hr' : LT.lt (↑r') (FormalMultilinearSeries.ofScalars E c).radius
    this : LT.lt 1 (HMul.hMul r' r)
    ⊢ Not (Summable fun n => HMul.hMul (Norm.norm (FormalMultilinearSeries.ofScala …
  -/
  apply not_summable_of_ratio_test_tendsto_gt_one this
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    inst✝ : NormOneClass E
    r : NNReal
    hr : Ne r 0
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    r' : NNReal
    hr' : LT.lt (↑r') (FormalMultilinearSeries.ofScalars E c).radius
    this : LT.lt 1 (HMul.hMul r' r)
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (HMul.hMul (Norm.norm (FormalM …
  -/
  simp_rw [ofScalars_norm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    inst✝ : NormOneClass E
    r : NNReal
    hr : Ne r 0
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    r' : NNReal
    hr' : LT.lt (↑r') (FormalMultilinearSeries.ofScalars E c).radius
    this : LT.lt 1 (HMul.hMul r' r)
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (HMul.hMul (Norm.norm (c (HAdd …
  -/
  exact tendsto_succ_norm_div_norm c (by aesop) hc
  /-
    🎉 no goals
  -/


/-- A convenience lemma restating the result of `ofScalars_radius_eq_inv_of_tendsto` under
the inverse ratio. -/
theorem ofScalars_radius_eq_of_tendsto [NormOneClass E] {r : NNReal} (hr : r ≠ 0)
    (hc : Tendsto (fun n ↦ ‖c n‖ / ‖c n.succ‖) atTop (𝓝 r)) :
      (ofScalars E c).radius = ofNNReal r := by
  suffices Tendsto (fun n ↦ ‖c n.succ‖ / ‖c n‖) atTop (𝓝 r⁻¹) by
    convert ofScalars_radius_eq_inv_of_tendsto E c (inv_ne_zero hr) this
    simp
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    inst✝ : NormOneClass E
    r : NNReal
    hr : Ne r 0
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n)) (Norm.norm (c n.succ …
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n))) …
  -/
  convert hc.inv₀ (NNReal.coe_ne_zero.mpr hr) using 1
  /-
    case h.e'_3
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    inst✝ : NormOneClass E
    r : NNReal
    hr : Ne r 0
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n)) (Norm.norm (c n.succ …
    ⊢ Eq (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n))) fun x => In …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The ratio test stating that if `‖c n.succ‖ / ‖c n‖` tends to zero, the radius is unbounded.
This requires that the coefficients are eventually non-zero as
`‖c n.succ‖ / 0 = 0` by convention. -/
theorem ofScalars_radius_eq_top_of_tendsto (hc : ∀ᶠ n in atTop, c n ≠ 0)
    (hc' : Tendsto (fun n ↦ ‖c n.succ‖ / ‖c n‖) atTop (𝓝 0)) : (ofScalars E c).radius = ⊤ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedRing E
    inst✝ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    hc : Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop
    hc' : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
    ⊢ Eq (FormalMultilinearSeries.ofScalars E c).radius Top.top
  -/
  refine radius_eq_top_of_summable_norm _ fun r' ↦ ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedRing E
    inst✝ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    hc : Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop
    hc' : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
    r' : NNReal
    ⊢ Summable fun n => HMul.hMul (Norm.norm (FormalMultilinearSeries.ofScalars E  …
  -/
  by_cases hrz : r' = 0
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedRing E
      inst✝ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      hc : Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop
      hc' : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
      r' : NNReal
      hrz : Eq r' 0
      ⊢ Summable fun n => HMul.hMul (Norm.norm (FormalMultilinearSeries.ofScalars E  …
    -/
  · apply Summable.comp_nat_add (k := 1)
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedRing E
      inst✝ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      hc : Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop
      hc' : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
      r' : NNReal
      hrz : Eq r' 0
      ⊢ Summable fun n => HMul.hMul (Norm.norm (FormalMultilinearSeries.ofScalars E  …
    -/
    simp [hrz]
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedRing E
      inst✝ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      hc : Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop
      hc' : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
      r' : NNReal
      hrz : Eq r' 0
      ⊢ Summable fun n => 0
    -/
    exact (summable_const_iff 0).mpr rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : NormedRing E
      inst✝ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      hc : Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop
      hc' : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
      r' : NNReal
      hrz : Not (Eq r' 0)
      ⊢ Summable fun n => HMul.hMul (Norm.norm (FormalMultilinearSeries.ofScalars E  …
    -/
  · refine Summable.of_norm_bounded_eventually (fun n ↦ ‖‖c n‖ * r' ^ n‖) ?_ ?_
      /-
        case neg.refine_1
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedRing E
        inst✝ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        hc : Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
        r' : NNReal
        hrz : Not (Eq r' 0)
        ⊢ Summable fun n => Norm.norm (HMul.hMul (Norm.norm (c n)) (HPow.hPow (↑r') n))
      -/
    · apply summable_of_ratio_test_tendsto_lt_one zero_lt_one (hc.mp (Eventually.of_forall ?_))
        /-
          case neg.refine_1
          𝕜 : Type u_1
          E : Type u_2
          inst✝² : NontriviallyNormedField 𝕜
          inst✝¹ : NormedRing E
          inst✝ : NormedAlgebra 𝕜 E
          c : Nat → 𝕜
          hc : Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop
          hc' : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
          r' : NNReal
          hrz : Not (Eq r' 0)
          ⊢ Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (Norm.norm (HMul.hMul (Norm.no …
        -/
      · simp only [norm_norm]
        /-
          case neg.refine_1
          𝕜 : Type u_1
          E : Type u_2
          inst✝² : NontriviallyNormedField 𝕜
          inst✝¹ : NormedRing E
          inst✝ : NormedAlgebra 𝕜 E
          c : Nat → 𝕜
          hc : Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop
          hc' : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
          r' : NNReal
          hrz : Not (Eq r' 0)
          ⊢ Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (HMul.hMul (Norm.norm (c (HAdd …
        -/
        exact mul_zero (_ : ℝ) ▸ tendsto_succ_norm_div_norm _ hrz (NNReal.coe_zero ▸ hc')
        /-
          🎉 no goals
        -/
        /-
          𝕜 : Type u_1
          E : Type u_2
          inst✝² : NontriviallyNormedField 𝕜
          inst✝¹ : NormedRing E
          inst✝ : NormedAlgebra 𝕜 E
          c : Nat → 𝕜
          hc : Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop
          hc' : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
          r' : NNReal
          hrz : Not (Eq r' 0)
          ⊢ ∀ (x : Nat), Ne (c x) 0 → Ne (Norm.norm (HMul.hMul (Norm.norm (c x)) (HPow.h …
        -/
      · aesop
        /-
          🎉 no goals
        -/
      /-
        case neg.refine_2
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedRing E
        inst✝ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        hc : Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
        r' : NNReal
        hrz : Not (Eq r' 0)
        ⊢ Filter.Eventually (fun i => LE.le (Norm.norm (HMul.hMul (Norm.norm (FormalMu …
      -/
    · filter_upwards [eventually_cofinite_ne 0] with n hn
      /-
        case h
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedRing E
        inst✝ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        hc : Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
        r' : NNReal
        hrz : Not (Eq r' 0)
        n : Nat
        hn : Ne n 0
        ⊢ LE.le (Norm.norm (HMul.hMul (Norm.norm (FormalMultilinearSeries.ofScalars E  …
      -/
      simp only [norm_mul, norm_norm, norm_pow, NNReal.norm_eq]
      /-
        case h
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedRing E
        inst✝ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        hc : Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
        r' : NNReal
        hrz : Not (Eq r' 0)
        n : Nat
        hn : Ne n 0
        ⊢ LE.le (HMul.hMul (Norm.norm (FormalMultilinearSeries.ofScalars E c n)) (HPow …
      -/
      gcongr
      /-
        case h.h
        𝕜 : Type u_1
        E : Type u_2
        inst✝² : NontriviallyNormedField 𝕜
        inst✝¹ : NormedRing E
        inst✝ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        hc : Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
        r' : NNReal
        hrz : Not (Eq r' 0)
        n : Nat
        hn : Ne n 0
        ⊢ LE.le (Norm.norm (FormalMultilinearSeries.ofScalars E c n)) (Norm.norm (c n))
      -/
      exact ofScalars_norm_le E c n (Nat.pos_iff_ne_zero.mpr hn)
      /-
        🎉 no goals
      -/


/-- If `‖c n.succ‖ / ‖c n‖` is unbounded, then the radius of convergence is zero. -/
theorem ofScalars_radius_eq_zero_of_tendsto [NormOneClass E]
    (hc : Tendsto (fun n ↦ ‖c n.succ‖ / ‖c n‖) atTop atTop) : (ofScalars E c).radius = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    inst✝ : NormOneClass E
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    ⊢ Eq (FormalMultilinearSeries.ofScalars E c).radius 0
  -/
  suffices (ofScalars E c).radius ≤ 0 by aesop
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    inst✝ : NormOneClass E
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    ⊢ LE.le (FormalMultilinearSeries.ofScalars E c).radius 0
  -/
  refine le_of_forall_nnreal_lt (fun r hr ↦ ?_)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    inst✝ : NormOneClass E
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    r : NNReal
    hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
    ⊢ LE.le (↑r) 0
  -/
  rw [← coe_zero, coe_le_coe]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    inst✝ : NormOneClass E
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    r : NNReal
    hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
    ⊢ LE.le r 0
  -/
  have := FormalMultilinearSeries.summable_norm_mul_pow _ hr
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    inst✝ : NormOneClass E
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    r : NNReal
    hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
    this : Summable fun n => HMul.hMul (Norm.norm (FormalMultilinearSeries.ofScala …
    ⊢ LE.le r 0
  -/
  contrapose! this
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    inst✝ : NormOneClass E
    hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
    r : NNReal
    hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
    this : LT.lt 0 r
    ⊢ Not (Summable fun n => HMul.hMul (Norm.norm (FormalMultilinearSeries.ofScala …
  -/
  apply not_summable_of_ratio_norm_eventually_ge one_lt_two
    /-
      case hf
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
      r : NNReal
      hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
      this : LT.lt 0 r
      ⊢ Filter.Frequently (fun n => Ne (Norm.norm (HMul.hMul (Norm.norm (FormalMulti …
    -/
  · contrapose! hc
    /-
      case hf
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      r : NNReal
      hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
      this : LT.lt 0 r
      hc : Not (Filter.Frequently (fun n => Ne (Norm.norm (HMul.hMul (Norm.norm (For …
      ⊢ Not (Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c …
    -/
    apply not_tendsto_atTop_of_tendsto_nhds (a:=0)
    /-
      case hf
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      r : NNReal
      hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
      this : LT.lt 0 r
      hc : Not (Filter.Frequently (fun n => Ne (Norm.norm (HMul.hMul (Norm.norm (For …
      ⊢ Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n))) …
    -/
    rw [not_frequently] at hc
    /-
      case hf
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      r : NNReal
      hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
      this : LT.lt 0 r
      hc : Filter.Eventually (fun x => Not (Ne (Norm.norm (HMul.hMul (Norm.norm (For …
      ⊢ Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n))) …
    -/
    apply Tendsto.congr' ?_ tendsto_const_nhds
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      r : NNReal
      hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
      this : LT.lt 0 r
      hc : Filter.Eventually (fun x => Not (Ne (Norm.norm (HMul.hMul (Norm.norm (For …
      ⊢ Filter.atTop.EventuallyEq (fun x => 0) fun n => HDiv.hDiv (Norm.norm (c n.su …
    -/
    filter_upwards [hc] with n hc'
    /-
      case h
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      r : NNReal
      hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
      this : LT.lt 0 r
      hc : Filter.Eventually (fun x => Not (Ne (Norm.norm (HMul.hMul (Norm.norm (For …
      n : Nat
      hc' : Not (Ne (Norm.norm (HMul.hMul (Norm.norm (FormalMultilinearSeries.ofScal …
      ⊢ Eq 0 (HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n)))
    -/
    rw [ofScalars_norm, norm_mul, norm_norm, not_ne_iff, mul_eq_zero] at hc'
    /-
      case h
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      r : NNReal
      hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
      this : LT.lt 0 r
      hc : Filter.Eventually (fun x => Not (Ne (Norm.norm (HMul.hMul (Norm.norm (For …
      n : Nat
      hc' : Or (Eq (Norm.norm (c n)) 0) (Eq (Norm.norm (HPow.hPow (↑r) n)) 0)
      ⊢ Eq 0 (HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n)))
    -/
                  /-
                    🎉 no goals
                  -/
    cases hc' <;> aesop
                  /-
                    🎉 no goals
                  -/
    /-
      case h
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      hc : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n …
      r : NNReal
      hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
      this : LT.lt 0 r
      ⊢ Filter.Eventually (fun n => LE.le (HMul.hMul 2 (Norm.norm (HMul.hMul (Norm.n …
    -/
  · filter_upwards [hc.eventually_ge_atTop (2*r⁻¹), eventually_ne_atTop 0] with n hc hn
    /-
      case h
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      hc✝ : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
      r : NNReal
      hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
      this : LT.lt 0 r
      n : Nat
      hc : LE.le (HMul.hMul 2 ↑(Inv.inv r)) (HDiv.hDiv (Norm.norm (c n.succ)) (Norm. …
      hn : Ne n 0
      ⊢ LE.le (HMul.hMul 2 (Norm.norm (HMul.hMul (Norm.norm (FormalMultilinearSeries …
    -/
    simp only [ofScalars_norm, norm_mul, norm_norm, norm_pow, NNReal.norm_eq]
    /-
      case h
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      hc✝ : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
      r : NNReal
      hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
      this : LT.lt 0 r
      n : Nat
      hc : LE.le (HMul.hMul 2 ↑(Inv.inv r)) (HDiv.hDiv (Norm.norm (c n.succ)) (Norm. …
      hn : Ne n 0
      ⊢ LE.le (HMul.hMul 2 (HMul.hMul (Norm.norm (c n)) (HPow.hPow (↑r) n))) (HMul.h …
    -/
    rw [mul_comm ‖c n‖, ← mul_assoc, ← div_le_div_iff₀, mul_div_assoc]
      /-
        case h
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        hc✝ : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
        r : NNReal
        hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
        this : LT.lt 0 r
        n : Nat
        hc : LE.le (HMul.hMul 2 ↑(Inv.inv r)) (HDiv.hDiv (Norm.norm (c n.succ)) (Norm. …
        hn : Ne n 0
        ⊢ LE.le (HMul.hMul 2 (HDiv.hDiv (HPow.hPow (↑r) n) (HPow.hPow (↑r) (HAdd.hAdd  …
      -/
    · convert hc
      /-
        case h.e'_3.h.e'_6
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        hc✝ : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
        r : NNReal
        hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
        this : LT.lt 0 r
        n : Nat
        hc : LE.le (HMul.hMul 2 ↑(Inv.inv r)) (HDiv.hDiv (Norm.norm (c n.succ)) (Norm. …
        hn : Ne n 0
        ⊢ Eq (HDiv.hDiv (HPow.hPow (↑r) n) (HPow.hPow (↑r) (HAdd.hAdd n 1))) ↑(Inv.inv …
      -/
      rw [pow_succ, div_mul_cancel_left₀, NNReal.coe_inv]
      /-
        case h.e'_3.h.e'_6.ha
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        hc✝ : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
        r : NNReal
        hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
        this : LT.lt 0 r
        n : Nat
        hc : LE.le (HMul.hMul 2 ↑(Inv.inv r)) (HDiv.hDiv (Norm.norm (c n.succ)) (Norm. …
        hn : Ne n 0
        ⊢ Ne (HPow.hPow (↑r) n) 0
      -/
      aesop
      /-
        🎉 no goals
      -/
      /-
        case h.hb
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        hc✝ : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
        r : NNReal
        hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
        this : LT.lt 0 r
        n : Nat
        hc : LE.le (HMul.hMul 2 ↑(Inv.inv r)) (HDiv.hDiv (Norm.norm (c n.succ)) (Norm. …
        hn : Ne n 0
        ⊢ LT.lt 0 (HPow.hPow (↑r) (HAdd.hAdd n 1))
      -/
    · aesop
      /-
        🎉 no goals
      -/
      /-
        case h.hd
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        hc✝ : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
        r : NNReal
        hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
        this : LT.lt 0 r
        n : Nat
        hc : LE.le (HMul.hMul 2 ↑(Inv.inv r)) (HDiv.hDiv (Norm.norm (c n.succ)) (Norm. …
        hn : Ne n 0
        ⊢ LT.lt 0 (Norm.norm (c n))
      -/
    · refine Ne.lt_of_le (fun hr' ↦ Not.elim ?_ hc) (norm_nonneg _)
      /-
        case h.hd
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        hc✝ : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
        r : NNReal
        hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
        this : LT.lt 0 r
        n : Nat
        hc : LE.le (HMul.hMul 2 ↑(Inv.inv r)) (HDiv.hDiv (Norm.norm (c n.succ)) (Norm. …
        hn : Ne n 0
        hr' : Eq 0 (Norm.norm (c n))
        ⊢ Not (LE.le (HMul.hMul 2 ↑(Inv.inv r)) (HDiv.hDiv (Norm.norm (c n.succ)) (Nor …
      -/
      rw [← hr']
      /-
        case h.hd
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        hc✝ : Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c  …
        r : NNReal
        hr : LT.lt (↑r) (FormalMultilinearSeries.ofScalars E c).radius
        this : LT.lt 0 r
        n : Nat
        hc : LE.le (HMul.hMul 2 ↑(Inv.inv r)) (HDiv.hDiv (Norm.norm (c n.succ)) (Norm. …
        hn : Ne n 0
        hr' : Eq 0 (Norm.norm (c n))
        ⊢ Not (LE.le (HMul.hMul 2 ↑(Inv.inv r)) (HDiv.hDiv (Norm.norm (c n.succ)) 0))
      -/
      simp [this]
      /-
        🎉 no goals
      -/


/-- This theorem combines the results of the special cases above, using `ENNReal` division to remove
the requirement that the ratio is eventually non-zero. -/
theorem ofScalars_radius_eq_inv_of_tendsto_ENNReal [NormOneClass E] {r : ℝ≥0∞}
    (hc' : Tendsto (fun n ↦ ENNReal.ofReal ‖c n.succ‖ / ENNReal.ofReal ‖c n‖) atTop (𝓝 r)) :
      (ofScalars E c).radius = r⁻¹ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    c : Nat → 𝕜
    inst✝ : NormOneClass E
    r : ENNReal
    hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
    ⊢ Eq (FormalMultilinearSeries.ofScalars E c).radius (Inv.inv r)
  -/
  rcases ENNReal.trichotomy r with (hr | hr | hr)
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      r : ENNReal
      hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
      hr : Eq r 0
      ⊢ Eq (FormalMultilinearSeries.ofScalars E c).radius (Inv.inv r)
    -/
  · simp_rw [hr, inv_zero] at hc' ⊢
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      r : ENNReal
      hr : Eq r 0
      hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
      ⊢ Eq (FormalMultilinearSeries.ofScalars E c).radius Top.top
    -/
    by_cases h : (∀ᶠ (n : ℕ) in atTop, c n ≠ 0)
      /-
        case pos
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        r : ENNReal
        hr : Eq r 0
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
        h : Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop
        ⊢ Eq (FormalMultilinearSeries.ofScalars E c).radius Top.top
      -/
    · apply ofScalars_radius_eq_top_of_tendsto E c h ?_
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        r : ENNReal
        hr : Eq r 0
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
        h : Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop
        ⊢ Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n))) …
      -/
      refine Tendsto.congr' ?_ <| (tendsto_toReal zero_ne_top).comp hc'
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        r : ENNReal
        hr : Eq r 0
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
        h : Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop
        ⊢ Filter.atTop.EventuallyEq (Function.comp ENNReal.toReal fun n => HDiv.hDiv ( …
      -/
      filter_upwards [h]
      /-
        case h
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        r : ENNReal
        hr : Eq r 0
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
        h : Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop
        ⊢ ∀ (a : Nat), Ne (c a) 0 → Eq (Function.comp ENNReal.toReal (fun n => HDiv.hD …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        r : ENNReal
        hr : Eq r 0
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
        h : Not (Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop)
        ⊢ Eq (FormalMultilinearSeries.ofScalars E c).radius Top.top
      -/
    · apply (ofScalars E c).radius_eq_top_of_eventually_eq_zero
      /-
        case neg
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        r : ENNReal
        hr : Eq r 0
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
        h : Not (Filter.Eventually (fun n => Ne (c n) 0) Filter.atTop)
        ⊢ Filter.Eventually (fun n => Eq (FormalMultilinearSeries.ofScalars E c n) 0)  …
      -/
      simp only [eventually_atTop, not_exists, not_forall, Classical.not_imp, not_not] at h ⊢
      /-
        case neg
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        r : ENNReal
        hr : Eq r 0
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
        h : ∀ (x : Nat), Exists fun x_1 => Exists fun h => Eq (c x_1) 0
        ⊢ Exists fun a => ∀ (b : Nat), GE.ge b a → Eq (FormalMultilinearSeries.ofScala …
      -/
      obtain ⟨ti, hti⟩ := eventually_atTop.mp (hc'.eventually_ne zero_ne_top)
      /-
        case neg.intro
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        r : ENNReal
        hr : Eq r 0
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
        h : ∀ (x : Nat), Exists fun x_1 => Exists fun h => Eq (c x_1) 0
        ti : Nat
        hti : ∀ (b : Nat), GE.ge b ti → Ne (HDiv.hDiv (ENNReal.ofReal (Norm.norm (c b. …
        ⊢ Exists fun a => ∀ (b : Nat), GE.ge b a → Eq (FormalMultilinearSeries.ofScala …
      -/
      obtain ⟨zi, hzi, z⟩ := h ti
      /-
        case neg.intro.intro.intro
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        r : ENNReal
        hr : Eq r 0
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
        h : ∀ (x : Nat), Exists fun x_1 => Exists fun h => Eq (c x_1) 0
        ti : Nat
        hti : ∀ (b : Nat), GE.ge b ti → Ne (HDiv.hDiv (ENNReal.ofReal (Norm.norm (c b. …
        zi : Nat
        hzi : GE.ge zi ti
        z : Eq (c zi) 0
        ⊢ Exists fun a => ∀ (b : Nat), GE.ge b a → Eq (FormalMultilinearSeries.ofScala …
      -/
      refine ⟨zi, Nat.le_induction (ofScalars_eq_zero_of_scalar_zero E z) fun n hmn a ↦ ?_⟩
      /-
        case neg.intro.intro.intro
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        r : ENNReal
        hr : Eq r 0
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
        h : ∀ (x : Nat), Exists fun x_1 => Exists fun h => Eq (c x_1) 0
        ti : Nat
        hti : ∀ (b : Nat), GE.ge b ti → Ne (HDiv.hDiv (ENNReal.ofReal (Norm.norm (c b. …
        zi : Nat
        hzi : GE.ge zi ti
        z : Eq (c zi) 0
        n : Nat
        hmn : LE.le zi n
        a : Eq (FormalMultilinearSeries.ofScalars E c n) 0
        ⊢ Eq (FormalMultilinearSeries.ofScalars E c (HAdd.hAdd n 1)) 0
      -/
      nontriviality E
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        r : ENNReal
        hr : Eq r 0
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
        h : ∀ (x : Nat), Exists fun x_1 => Exists fun h => Eq (c x_1) 0
        ti : Nat
        hti : ∀ (b : Nat), GE.ge b ti → Ne (HDiv.hDiv (ENNReal.ofReal (Norm.norm (c b. …
        zi : Nat
        hzi : GE.ge zi ti
        z : Eq (c zi) 0
        n : Nat
        hmn : LE.le zi n
        a : Eq (FormalMultilinearSeries.ofScalars E c n) 0
        a✝ : Nontrivial E
        ⊢ Eq (FormalMultilinearSeries.ofScalars E c (HAdd.hAdd n 1)) 0
      -/
      simp only [ofScalars_eq_zero] at a ⊢
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        r : ENNReal
        hr : Eq r 0
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
        h : ∀ (x : Nat), Exists fun x_1 => Exists fun h => Eq (c x_1) 0
        ti : Nat
        hti : ∀ (b : Nat), GE.ge b ti → Ne (HDiv.hDiv (ENNReal.ofReal (Norm.norm (c b. …
        zi : Nat
        hzi : GE.ge zi ti
        z : Eq (c zi) 0
        n : Nat
        hmn : LE.le zi n
        a✝ : Nontrivial E
        a : Eq (c n) 0
        ⊢ Eq (c (HAdd.hAdd n 1)) 0
      -/
      contrapose! hti
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        r : ENNReal
        hr : Eq r 0
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
        h : ∀ (x : Nat), Exists fun x_1 => Exists fun h => Eq (c x_1) 0
        ti zi : Nat
        hzi : GE.ge zi ti
        z : Eq (c zi) 0
        n : Nat
        hmn : LE.le zi n
        a✝ : Nontrivial E
        a : Eq (c n) 0
        hti : Ne (c (HAdd.hAdd n 1)) 0
        ⊢ Exists fun b => And (GE.ge b ti) (Eq (HDiv.hDiv (ENNReal.ofReal (Norm.norm ( …
      -/
      exact ⟨n, hzi.trans hmn, ENNReal.div_eq_top.mpr (by simp [a, hti])⟩
      /-
        🎉 no goals
      -/
    /-
      case inr.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      r : ENNReal
      hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
      hr : Eq r Top.top
      ⊢ Eq (FormalMultilinearSeries.ofScalars E c).radius (Inv.inv r)
    -/
  · simp_rw [hr, inv_top] at hc' ⊢
    /-
      case inr.inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      r : ENNReal
      hr : Eq r Top.top
      hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
      ⊢ Eq (FormalMultilinearSeries.ofScalars E c).radius 0
    -/
    apply ofScalars_radius_eq_zero_of_tendsto E c ((tendsto_add_atTop_iff_nat 1).mp ?_)
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      r : ENNReal
      hr : Eq r Top.top
      hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
      ⊢ Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c (HAdd.hAdd n 1).succ)) (Nor …
    -/
    refine tendsto_ofReal_nhds_top.mp (Tendsto.congr' ?_ ((tendsto_add_atTop_iff_nat 1).mpr hc'))
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      r : ENNReal
      hr : Eq r Top.top
      hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
      ⊢ Filter.atTop.EventuallyEq (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c  …
    -/
    filter_upwards [hc'.eventually_ne top_ne_zero] with n hn
    /-
      case h
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      r : ENNReal
      hr : Eq r Top.top
      hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
      n : Nat
      hn : Ne (HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ))) (ENNReal.ofReal (No …
      ⊢ Eq (HDiv.hDiv (ENNReal.ofReal (Norm.norm (c (HAdd.hAdd n 1).succ))) (ENNReal …
    -/
    apply (ofReal_div_of_pos (Ne.lt_of_le (Ne.symm ?_) (norm_nonneg _))).symm
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      r : ENNReal
      hr : Eq r Top.top
      hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
      n : Nat
      hn : Ne (HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ))) (ENNReal.ofReal (No …
      ⊢ Ne (Norm.norm (c (HAdd.hAdd n 1))) 0
    -/
    aesop
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      r : ENNReal
      hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
      hr : LT.lt 0 r.toReal
      ⊢ Eq (FormalMultilinearSeries.ofScalars E c).radius (Inv.inv r)
    -/
  · have hr' := toReal_ne_zero.mp hr.ne.symm
    /-
      case inr.inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      r : ENNReal
      hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
      hr : LT.lt 0 r.toReal
      hr' : And (Ne r 0) (Ne r Top.top)
      ⊢ Eq (FormalMultilinearSeries.ofScalars E c).radius (Inv.inv r)
    -/
    have hr'' := toNNReal_ne_zero.mpr hr' -- this result could go in ENNReal
    /-
      case inr.inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : NormedRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      c : Nat → 𝕜
      inst✝ : NormOneClass E
      r : ENNReal
      hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
      hr : LT.lt 0 r.toReal
      hr' : And (Ne r 0) (Ne r Top.top)
      hr'' : Ne r.toNNReal 0
      ⊢ Eq (FormalMultilinearSeries.ofScalars E c).radius (Inv.inv r)
    -/
    convert ofScalars_radius_eq_inv_of_tendsto E c hr'' ?_
      /-
        case h.e'_3
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        r : ENNReal
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
        hr : LT.lt 0 r.toReal
        hr' : And (Ne r 0) (Ne r Top.top)
        hr'' : Ne r.toNNReal 0
        ⊢ Eq (Inv.inv r) ↑(Inv.inv r.toNNReal)
      -/
    · simp [ENNReal.coe_inv hr'', ENNReal.coe_toNNReal (toReal_ne_zero.mp hr.ne.symm).2]
      /-
        🎉 no goals
      -/
      /-
        case inr.inr
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        r : ENNReal
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
        hr : LT.lt 0 r.toReal
        hr' : And (Ne r 0) (Ne r Top.top)
        hr'' : Ne r.toNNReal 0
        ⊢ Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n))) …
      -/
    · simp_rw [ENNReal.coe_toNNReal_eq_toReal]
      /-
        case inr.inr
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        r : ENNReal
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
        hr : LT.lt 0 r.toReal
        hr' : And (Ne r 0) (Ne r Top.top)
        hr'' : Ne r.toNNReal 0
        ⊢ Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm (c n.succ)) (Norm.norm (c n))) …
      -/
      refine Tendsto.congr' ?_ <| (tendsto_toReal hr'.2).comp hc'
      /-
        case inr.inr
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        r : ENNReal
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
        hr : LT.lt 0 r.toReal
        hr' : And (Ne r 0) (Ne r Top.top)
        hr'' : Ne r.toNNReal 0
        ⊢ Filter.atTop.EventuallyEq (Function.comp ENNReal.toReal fun n => HDiv.hDiv ( …
      -/
      filter_upwards [hc'.eventually_ne hr'.1, hc'.eventually_ne hr'.2]
      /-
        case h
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : NontriviallyNormedField 𝕜
        inst✝² : NormedRing E
        inst✝¹ : NormedAlgebra 𝕜 E
        c : Nat → 𝕜
        inst✝ : NormOneClass E
        r : ENNReal
        hc' : Filter.Tendsto (fun n => HDiv.hDiv (ENNReal.ofReal (Norm.norm (c n.succ) …
        hr : LT.lt 0 r.toReal
        hr' : And (Ne r 0) (Ne r Top.top)
        hr'' : Ne r.toNNReal 0
        ⊢ ∀ (a : Nat), Ne (HDiv.hDiv (ENNReal.ofReal (Norm.norm (c a.succ))) (ENNReal. …
      -/
      simp
      /-
        🎉 no goals
      -/



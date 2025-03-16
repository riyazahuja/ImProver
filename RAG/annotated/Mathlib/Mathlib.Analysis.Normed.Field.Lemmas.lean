theorem Filter.Tendsto.zero_mul_isBoundedUnder_le {f g : ι → α} {l : Filter ι}
    (hf : Tendsto f l (𝓝 0)) (hg : IsBoundedUnder (· ≤ ·) l ((‖·‖) ∘ g)) :
    Tendsto (fun x => f x * g x) l (𝓝 0) :=
  hf.op_zero_isBoundedUnder_le hg (· * ·) norm_mul_le


theorem Filter.isBoundedUnder_le_mul_tendsto_zero {f g : ι → α} {l : Filter ι}
    (hf : IsBoundedUnder (· ≤ ·) l (norm ∘ f)) (hg : Tendsto g l (𝓝 0)) :
    Tendsto (fun x => f x * g x) l (𝓝 0) :=
  hg.op_zero_isBoundedUnder_le hf (flip (· * ·)) fun x y =>
    (norm_mul_le y x).trans_eq (mul_comm _ _)


/-- Non-unital seminormed ring structure on the product of finitely many non-unital seminormed
rings, using the sup norm. -/
instance Pi.nonUnitalSeminormedRing {π : ι → Type*} [Fintype ι]
    [∀ i, NonUnitalSeminormedRing (π i)] : NonUnitalSeminormedRing (∀ i, π i) :=
  { Pi.seminormedAddCommGroup, Pi.nonUnitalRing with
    norm_mul := fun x y =>
      NNReal.coe_mono <|
        calc
          (Finset.univ.sup fun i => ‖x i * y i‖₊) ≤
              Finset.univ.sup ((fun i => ‖x i‖₊) * fun i => ‖y i‖₊) :=
            Finset.sup_mono_fun fun _ _ => norm_mul_le _ _
          _ ≤ (Finset.univ.sup fun i => ‖x i‖₊) * Finset.univ.sup fun i => ‖y i‖₊ :=
            Finset.sup_mul_le_mul_sup_of_nonneg (fun _ _ => zero_le _) fun _ _ => zero_le _
           }


/-- Seminormed ring structure on the product of finitely many seminormed rings,
  using the sup norm. -/
instance Pi.seminormedRing {π : ι → Type*} [Fintype ι] [∀ i, SeminormedRing (π i)] :
    SeminormedRing (∀ i, π i) :=
  { Pi.nonUnitalSeminormedRing, Pi.ring with }


/-- Normed ring structure on the product of finitely many non-unital normed rings, using the sup
norm. -/
instance Pi.nonUnitalNormedRing {π : ι → Type*} [Fintype ι] [∀ i, NonUnitalNormedRing (π i)] :
    NonUnitalNormedRing (∀ i, π i) :=
  { Pi.nonUnitalSeminormedRing, Pi.normedAddCommGroup with }


/-- Normed ring structure on the product of finitely many normed rings, using the sup norm. -/
instance Pi.normedRing {π : ι → Type*} [Fintype ι] [∀ i, NormedRing (π i)] :
    NormedRing (∀ i, π i) :=
  { Pi.seminormedRing, Pi.normedAddCommGroup with }


/-- Non-unital seminormed commutative ring structure on the product of finitely many non-unital
seminormed commutative rings, using the sup norm. -/
instance Pi.nonUnitalSeminormedCommRing {π : ι → Type*} [Fintype ι]
    [∀ i, NonUnitalSeminormedCommRing (π i)] : NonUnitalSeminormedCommRing (∀ i, π i) :=
  { Pi.nonUnitalSeminormedRing, Pi.nonUnitalCommRing with }


/-- Normed commutative ring structure on the product of finitely many non-unital normed
commutative rings, using the sup norm. -/
instance Pi.nonUnitalNormedCommRing {π : ι → Type*} [Fintype ι]
    [∀ i, NonUnitalNormedCommRing (π i)] : NonUnitalNormedCommRing (∀ i, π i) :=
  { Pi.nonUnitalSeminormedCommRing, Pi.normedAddCommGroup with }


/-- Seminormed commutative ring structure on the product of finitely many seminormed commutative
rings, using the sup norm. -/
instance Pi.seminormedCommRing {π : ι → Type*} [Fintype ι] [∀ i, SeminormedCommRing (π i)] :
    SeminormedCommRing (∀ i, π i) :=
  { Pi.nonUnitalSeminormedCommRing, Pi.ring with }


/-- Normed commutative ring structure on the product of finitely many normed commutative rings,
using the sup norm. -/
instance Pi.normedCommutativeRing {π : ι → Type*} [Fintype ι] [∀ i, NormedCommRing (π i)] :
    NormedCommRing (∀ i, π i) :=
  { Pi.seminormedCommRing, Pi.normedAddCommGroup with }


instance (priority := 100) NonUnitalSeminormedRing.toContinuousMul [NonUnitalSeminormedRing α] :
    ContinuousMul α :=
  ⟨continuous_iff_continuousAt.2 fun x =>
      tendsto_iff_norm_sub_tendsto_zero.2 <| by
        have : ∀ e : α × α,
            ‖e.1 * e.2 - x.1 * x.2‖ ≤ ‖e.1‖ * ‖e.2 - x.2‖ + ‖e.1 - x.1‖ * ‖x.2‖ := by
          intro e
          calc
            ‖e.1 * e.2 - x.1 * x.2‖ ≤ ‖e.1 * (e.2 - x.2) + (e.1 - x.1) * x.2‖ := by
              rw [mul_sub, sub_mul, sub_add_sub_cancel]
            _ ≤ ‖e.1‖ * ‖e.2 - x.2‖ + ‖e.1 - x.1‖ * ‖x.2‖ :=
              norm_add_le_of_le (norm_mul_le _ _) (norm_mul_le _ _)
        /-
          α : Type u_1
          β : Type u_2
          ι : Type u_3
          inst✝ : NonUnitalSeminormedRing α
          x : Prod α α
          this : ∀ (e : Prod α α), LE.le (Norm.norm (HSub.hSub (HMul.hMul e.1 e.2) (HMul …
          ⊢ Filter.Tendsto (fun e => Norm.norm (HSub.hSub (HMul.hMul e.1 e.2) ((fun p => …
        -/
        refine squeeze_zero (fun e => norm_nonneg _) this ?_
        convert
          ((continuous_fst.tendsto x).norm.mul
                ((continuous_snd.tendsto x).sub tendsto_const_nhds).norm).add
            (((continuous_fst.tendsto x).sub tendsto_const_nhds).norm.mul _)
        -- Porting note: `show` used to select a goal to work on
        /-
          case h.e'_5.h.e'_3
          α : Type u_1
          β : Type u_2
          ι : Type u_3
          inst✝ : NonUnitalSeminormedRing α
          x : Prod α α
          this : ∀ (e : Prod α α), LE.le (Norm.norm (HSub.hSub (HMul.hMul e.1 e.2) (HMul …
          ⊢ Eq 0 (HAdd.hAdd (HMul.hMul (Norm.norm x.1) (Norm.norm (HSub.hSub x.2 x.2)))  …
        -/
        rotate_right
          /-
            case convert_5
            α : Type u_1
            β : Type u_2
            ι : Type u_3
            inst✝ : NonUnitalSeminormedRing α
            x : Prod α α
            this : ∀ (e : Prod α α), LE.le (Norm.norm (HSub.hSub (HMul.hMul e.1 e.2) (HMul …
            ⊢ Filter.Tendsto (fun t => Norm.norm x.2) (nhds x) (nhds ?convert_4)
          -/
        · show Tendsto _ _ _
          /-
            case convert_5
            α : Type u_1
            β : Type u_2
            ι : Type u_3
            inst✝ : NonUnitalSeminormedRing α
            x : Prod α α
            this : ∀ (e : Prod α α), LE.le (Norm.norm (HSub.hSub (HMul.hMul e.1 e.2) (HMul …
            ⊢ Filter.Tendsto (fun t => Norm.norm x.2) (nhds x) (nhds ?convert_4)
          -/
          exact tendsto_const_nhds
          /-
            🎉 no goals
          -/
          /-
            case h.e'_5.h.e'_3
            α : Type u_1
            β : Type u_2
            ι : Type u_3
            inst✝ : NonUnitalSeminormedRing α
            x : Prod α α
            this : ∀ (e : Prod α α), LE.le (Norm.norm (HSub.hSub (HMul.hMul e.1 e.2) (HMul …
            ⊢ Eq 0 (HAdd.hAdd (HMul.hMul (Norm.norm x.1) (Norm.norm (HSub.hSub x.2 x.2)))  …
          -/
        · simp⟩
          /-
            🎉 no goals
          -/

-- see Note [lower instance priority]

/-- A seminormed ring is a topological ring. -/
instance (priority := 100) NonUnitalSeminormedRing.toTopologicalRing [NonUnitalSeminormedRing α] :
    TopologicalRing α where


instance [NonUnitalSeminormedRing α] : NonUnitalNormedRing (SeparationQuotient α) where
  __ : NonUnitalRing (SeparationQuotient α) := inferInstance
  __ : NormedAddCommGroup (SeparationQuotient α) := inferInstance
  norm_mul := Quotient.ind₂ norm_mul_le


instance [NonUnitalSeminormedCommRing α] : NonUnitalNormedCommRing (SeparationQuotient α) where
  __ : NonUnitalCommRing (SeparationQuotient α) := inferInstance
  __ : NormedAddCommGroup (SeparationQuotient α) := inferInstance
  norm_mul := Quotient.ind₂ norm_mul_le


instance [SeminormedRing α] : NormedRing (SeparationQuotient α) where
  __ : Ring (SeparationQuotient α) := inferInstance
  __ : NormedAddCommGroup (SeparationQuotient α) := inferInstance
  norm_mul := Quotient.ind₂ norm_mul_le


instance [SeminormedCommRing α] : NormedCommRing (SeparationQuotient α) where
  __ : CommRing (SeparationQuotient α) := inferInstance
  __ : NormedAddCommGroup (SeparationQuotient α) := inferInstance
  norm_mul := Quotient.ind₂ norm_mul_le


instance [SeminormedAddCommGroup α] [One α] [NormOneClass α] :
    NormOneClass (SeparationQuotient α) where
  norm_one := norm_one (α := α)


lemma antilipschitzWith_mul_left {a : α} (ha : a ≠ 0) : AntilipschitzWith (‖a‖₊⁻¹) (a * ·) :=
                                                /-
                                                  α : Type u_1
                                                  inst✝ : NormedDivisionRing α
                                                  a : α
                                                  ha : Ne a 0
                                                  x✝¹ x✝ : α
                                                  ⊢ LE.le (Dist.dist x✝¹ x✝) (HMul.hMul (↑(Inv.inv (NNNorm.nnnorm a))) (Dist.dis …
                                                -/
  AntilipschitzWith.of_le_mul_dist fun _ _ ↦ by simp [dist_eq_norm, ← _root_.mul_sub, ha]
                                                /-
                                                  🎉 no goals
                                                -/


lemma antilipschitzWith_mul_right {a : α} (ha : a ≠ 0) : AntilipschitzWith (‖a‖₊⁻¹) (· * a) :=
  AntilipschitzWith.of_le_mul_dist fun _ _ ↦ by
    /-
      α : Type u_1
      inst✝ : NormedDivisionRing α
      a : α
      ha : Ne a 0
      x✝¹ x✝ : α
      ⊢ LE.le (Dist.dist x✝¹ x✝) (HMul.hMul (↑(Inv.inv (NNNorm.nnnorm a))) (Dist.dis …
    -/
    simp [dist_eq_norm, ← _root_.sub_mul, ← mul_comm (‖a‖), ha]
    /-
      🎉 no goals
    -/


/-- Multiplication by a nonzero element `a` on the left
as a `DilationEquiv` of a normed division ring. -/
@[simps!]
def DilationEquiv.mulLeft (a : α) (ha : a ≠ 0) : α ≃ᵈ α where
  toEquiv := Equiv.mulLeft₀ a ha
  edist_eq' := ⟨‖a‖₊, nnnorm_ne_zero_iff.2 ha, fun x y ↦ by
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : NormedDivisionRing α
      a✝ a : α
      ha : Ne a 0
      x y : α
      ⊢ Eq (EDist.edist ((Equiv.mulLeft₀ a ha).toFun x) ((Equiv.mulLeft₀ a ha).toFun …
    -/
    simp [edist_nndist, nndist_eq_nnnorm, ← mul_sub]⟩
    /-
      🎉 no goals
    -/


/-- Multiplication by a nonzero element `a` on the right
as a `DilationEquiv` of a normed division ring. -/
@[simps!]
def DilationEquiv.mulRight (a : α) (ha : a ≠ 0) : α ≃ᵈ α where
  toEquiv := Equiv.mulRight₀ a ha
  edist_eq' := ⟨‖a‖₊, nnnorm_ne_zero_iff.2 ha, fun x y ↦ by
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : NormedDivisionRing α
      a✝ a : α
      ha : Ne a 0
      x y : α
      ⊢ Eq (EDist.edist ((Equiv.mulRight₀ a ha).toFun x) ((Equiv.mulRight₀ a ha).toF …
    -/
    simp [edist_nndist, nndist_eq_nnnorm, ← sub_mul, ← mul_comm (‖a‖₊)]⟩
    /-
      🎉 no goals
    -/


@[simp]
lemma comap_mul_left_cobounded {a : α} (ha : a ≠ 0) :
    comap (a * ·) (cobounded α) = cobounded α :=
  Dilation.comap_cobounded (DilationEquiv.mulLeft a ha)


@[simp]
lemma map_mul_left_cobounded {a : α} (ha : a ≠ 0) :
    map (a * ·) (cobounded α) = cobounded α :=
  DilationEquiv.map_cobounded (DilationEquiv.mulLeft a ha)


@[simp]
lemma comap_mul_right_cobounded {a : α} (ha : a ≠ 0) :
    comap (· * a) (cobounded α) = cobounded α :=
  Dilation.comap_cobounded (DilationEquiv.mulRight a ha)


@[simp]
lemma map_mul_right_cobounded {a : α} (ha : a ≠ 0) :
    map (· * a) (cobounded α) = cobounded α :=
  DilationEquiv.map_cobounded (DilationEquiv.mulRight a ha)


/-- Multiplication on the left by a nonzero element of a normed division ring tends to infinity at
infinity. -/
theorem tendsto_mul_left_cobounded {a : α} (ha : a ≠ 0) :
    Tendsto (a * ·) (cobounded α) (cobounded α) :=
  (map_mul_left_cobounded ha).le


/-- Multiplication on the right by a nonzero element of a normed division ring tends to infinity at
infinity. -/
theorem tendsto_mul_right_cobounded {a : α} (ha : a ≠ 0) :
    Tendsto (· * a) (cobounded α) (cobounded α) :=
  (map_mul_right_cobounded ha).le


@[simp]
lemma inv_cobounded₀ : (cobounded α)⁻¹ = 𝓝[≠] 0 := by
  rw [← comap_norm_atTop, ← Filter.comap_inv, ← comap_norm_nhdsGT_zero, ← inv_atTop₀,
    ← Filter.comap_inv]
  /-
    α : Type u_1
    inst✝ : NormedDivisionRing α
    ⊢ Eq (Filter.comap Inv.inv (Filter.comap Norm.norm Filter.atTop)) (Filter.coma …
  -/
  simp only [comap_comap, Function.comp_def, norm_inv]
  /-
    🎉 no goals
  -/


@[simp]
lemma inv_nhdsWithin_ne_zero : (𝓝[≠] (0 : α))⁻¹ = cobounded α := by
  /-
    α : Type u_1
    inst✝ : NormedDivisionRing α
    ⊢ Eq (Inv.inv (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0)))) (Bornol …
  -/
  rw [← inv_cobounded₀, inv_inv]
  /-
    🎉 no goals
  -/


lemma tendsto_inv₀_cobounded' : Tendsto Inv.inv (cobounded α) (𝓝[≠] 0) :=
  inv_cobounded₀.le


theorem tendsto_inv₀_cobounded : Tendsto Inv.inv (cobounded α) (𝓝 0) :=
  tendsto_inv₀_cobounded'.mono_right inf_le_left


lemma tendsto_inv₀_nhdsWithin_ne_zero : Tendsto Inv.inv (𝓝[≠] 0) (cobounded α) :=
  inv_nhdsWithin_ne_zero.le


instance (priority := 100) NormedDivisionRing.to_hasContinuousInv₀ : HasContinuousInv₀ α := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : NormedDivisionRing α
    a : α
    ⊢ HasContinuousInv₀ α
  -/
  refine ⟨fun r r0 => tendsto_iff_norm_sub_tendsto_zero.2 ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : NormedDivisionRing α
    a r : α
    r0 : Ne r 0
    ⊢ Filter.Tendsto (fun e => Norm.norm (HSub.hSub (Inv.inv e) (Inv.inv r))) (nhd …
  -/
  have r0' : 0 < ‖r‖ := norm_pos_iff.2 r0
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : NormedDivisionRing α
    a r : α
    r0 : Ne r 0
    r0' : LT.lt 0 (Norm.norm r)
    ⊢ Filter.Tendsto (fun e => Norm.norm (HSub.hSub (Inv.inv e) (Inv.inv r))) (nhd …
  -/
  rcases exists_between r0' with ⟨ε, ε0, εr⟩
  have : ∀ᶠ e in 𝓝 r, ‖e⁻¹ - r⁻¹‖ ≤ ‖r - e‖ / ‖r‖ / ε := by
    filter_upwards [(isOpen_lt continuous_const continuous_norm).eventually_mem εr] with e he
    have e0 : e ≠ 0 := norm_pos_iff.1 (ε0.trans he)
    calc
      ‖e⁻¹ - r⁻¹‖ = ‖r‖⁻¹ * ‖r - e‖ * ‖e‖⁻¹ := by
        rw [← norm_inv, ← norm_inv, ← norm_mul, ← norm_mul, mul_sub, sub_mul,
          mul_assoc _ e, inv_mul_cancel₀ r0, mul_inv_cancel₀ e0, one_mul, mul_one]
      _ = ‖r - e‖ / ‖r‖ / ‖e‖ := by field_simp [mul_comm]
      _ ≤ ‖r - e‖ / ‖r‖ / ε := by gcongr
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : NormedDivisionRing α
    a r : α
    r0 : Ne r 0
    r0' : LT.lt 0 (Norm.norm r)
    ε : Real
    ε0 : LT.lt 0 ε
    εr : LT.lt ε (Norm.norm r)
    this : Filter.Eventually (fun e => LE.le (Norm.norm (HSub.hSub (Inv.inv e) (In …
    ⊢ Filter.Tendsto (fun e => Norm.norm (HSub.hSub (Inv.inv e) (Inv.inv r))) (nhd …
  -/
  refine squeeze_zero' (Eventually.of_forall fun _ => norm_nonneg _) this ?_
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : NormedDivisionRing α
    a r : α
    r0 : Ne r 0
    r0' : LT.lt 0 (Norm.norm r)
    ε : Real
    ε0 : LT.lt 0 ε
    εr : LT.lt ε (Norm.norm r)
    this : Filter.Eventually (fun e => LE.le (Norm.norm (HSub.hSub (Inv.inv e) (In …
    ⊢ Filter.Tendsto (fun t => HDiv.hDiv (HDiv.hDiv (Norm.norm (HSub.hSub r t)) (N …
  -/
  refine (((continuous_const.sub continuous_id).norm.div_const _).div_const _).tendsto' _ _ ?_
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    inst✝ : NormedDivisionRing α
    a r : α
    r0 : Ne r 0
    r0' : LT.lt 0 (Norm.norm r)
    ε : Real
    ε0 : LT.lt 0 ε
    εr : LT.lt ε (Norm.norm r)
    this : Filter.Eventually (fun e => LE.le (Norm.norm (HSub.hSub (Inv.inv e) (In …
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv (Norm.norm (HSub.hSub r (id r))) (Norm.norm r)) ε) 0
  -/
  simp
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

/-- A normed division ring is a topological division ring. -/
instance (priority := 100) NormedDivisionRing.to_topologicalDivisionRing :
    TopologicalDivisionRing α where


protected lemma IsOfFinOrder.norm_eq_one (ha : IsOfFinOrder a) : ‖a‖ = 1 :=
  ((normHom : α →*₀ ℝ).toMonoidHom.isOfFinOrder ha).eq_one <| norm_nonneg _


@[simp] lemma AddChar.norm_apply {G : Type*} [AddLeftCancelMonoid G] [Finite G] (ψ : AddChar G α)
    (x : G) : ‖ψ x‖ = 1 := (ψ.toMonoidHom.isOfFinOrder <| isOfFinOrder_of_finite _).norm_eq_one


lemma NormedField.tendsto_norm_inv_nhdsNE_zero_atTop : Tendsto (fun x : α ↦ ‖x⁻¹‖) (𝓝[≠] 0) atTop :=
  (tendsto_inv_nhdsGT_zero.comp tendsto_norm_nhdsNE_zero).congr fun x ↦ (norm_inv x).symm


@[deprecated (since := "2024-12-22")]
alias NormedField.tendsto_norm_inverse_nhdsWithin_0_atTop :=
  NormedField.tendsto_norm_inv_nhdsNE_zero_atTop


lemma NormedField.tendsto_norm_zpow_nhdsNE_zero_atTop {m : ℤ} (hm : m < 0) :
    Tendsto (fun x : α ↦ ‖x ^ m‖) (𝓝[≠] 0) atTop := by
  /-
    α : Type u_1
    inst✝ : NormedDivisionRing α
    m : Int
    hm : LT.lt m 0
    ⊢ Filter.Tendsto (fun x => Norm.norm (HPow.hPow x m)) (nhdsWithin 0 (HasCompl. …
  -/
  obtain ⟨m, rfl⟩ := neg_surjective m
  /-
    case intro
    α : Type u_1
    inst✝ : NormedDivisionRing α
    m : Int
    hm : LT.lt (Neg.neg m) 0
    ⊢ Filter.Tendsto (fun x => Norm.norm (HPow.hPow x (Neg.neg m))) (nhdsWithin 0  …
  -/
  rw [neg_lt_zero] at hm
  /-
    case intro
    α : Type u_1
    inst✝ : NormedDivisionRing α
    m : Int
    hm : LT.lt 0 m
    ⊢ Filter.Tendsto (fun x => Norm.norm (HPow.hPow x (Neg.neg m))) (nhdsWithin 0  …
  -/
  lift m to ℕ using hm.le
  /-
    case intro.intro
    α : Type u_1
    inst✝ : NormedDivisionRing α
    m : Nat
    hm : LT.lt 0 ↑m
    ⊢ Filter.Tendsto (fun x => Norm.norm (HPow.hPow x (Neg.neg ↑m))) (nhdsWithin 0 …
  -/
  rw [Int.natCast_pos] at hm
  /-
    case intro.intro
    α : Type u_1
    inst✝ : NormedDivisionRing α
    m : Nat
    hm : LT.lt 0 m
    ⊢ Filter.Tendsto (fun x => Norm.norm (HPow.hPow x (Neg.neg ↑m))) (nhdsWithin 0 …
  -/
  simp only [norm_pow, zpow_neg, zpow_natCast, ← inv_pow]
  /-
    case intro.intro
    α : Type u_1
    inst✝ : NormedDivisionRing α
    m : Nat
    hm : LT.lt 0 m
    ⊢ Filter.Tendsto (fun x => HPow.hPow (Norm.norm (Inv.inv x)) m) (nhdsWithin 0  …
  -/
  exact (tendsto_pow_atTop hm.ne').comp NormedField.tendsto_norm_inv_nhdsNE_zero_atTop
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")]
alias NormedField.tendsto_norm_zpow_nhdsWithin_0_atTop :=
  NormedField.tendsto_norm_zpow_nhdsNE_zero_atTop


/-- A normed field is either nontrivially normed or has a discrete topology.
In the discrete topology case, all the norms are 1, by `norm_eq_one_iff_ne_zero_of_discrete`.
The nontrivially normed field instance is provided by a subtype with a proof that the
forgetful inheritance to the existing `NormedField` instance is definitionally true.
This allows one to have the new `NontriviallyNormedField` instance without data clashes. -/
lemma discreteTopology_or_nontriviallyNormedField (𝕜 : Type*) [h : NormedField 𝕜] :
    DiscreteTopology 𝕜 ∨ Nonempty ({h' : NontriviallyNormedField 𝕜 // h'.toNormedField = h}) := by
  /-
    𝕜 : Type u_4
    h : NormedField 𝕜
    ⊢ Or (DiscreteTopology 𝕜) (Nonempty (Subtype fun h' => Eq NontriviallyNormedFi …
  -/
  by_cases H : ∃ x : 𝕜, x ≠ 0 ∧ ‖x‖ ≠ 1
    /-
      case pos
      𝕜 : Type u_4
      h : NormedField 𝕜
      H : Exists fun x => And (Ne x 0) (Ne (Norm.norm x) 1)
      ⊢ Or (DiscreteTopology 𝕜) (Nonempty (Subtype fun h' => Eq NontriviallyNormedFi …
    -/
  · exact Or.inr ⟨(⟨NontriviallyNormedField.ofNormNeOne H, rfl⟩)⟩
    /-
      🎉 no goals
    -/
  · simp_rw [discreteTopology_iff_isOpen_singleton_zero, Metric.isOpen_singleton_iff, dist_eq_norm,
             sub_zero]
    /-
      case neg
      𝕜 : Type u_4
      h : NormedField 𝕜
      H : Not (Exists fun x => And (Ne x 0) (Ne (Norm.norm x) 1))
      ⊢ Or (Exists fun ε => And (GT.gt ε 0) (∀ (y : 𝕜), LT.lt (Norm.norm y) ε → Eq y …
    -/
    refine Or.inl ⟨1, zero_lt_one, ?_⟩
    /-
      case neg
      𝕜 : Type u_4
      h : NormedField 𝕜
      H : Not (Exists fun x => And (Ne x 0) (Ne (Norm.norm x) 1))
      ⊢ ∀ (y : 𝕜), LT.lt (Norm.norm y) 1 → Eq y 0
    -/
    contrapose! H
    /-
      case neg
      𝕜 : Type u_4
      h : NormedField 𝕜
      H : Exists fun y => And (LT.lt (Norm.norm y) 1) (Ne y 0)
      ⊢ Exists fun x => And (Ne x 0) (Ne (Norm.norm x) 1)
    -/
    refine H.imp ?_
    -- contextual to reuse the `a ≠ 0` hypothesis in the proof of `a ≠ 0 ∧ ‖a‖ ≠ 1`
    /-
      case neg
      𝕜 : Type u_4
      h : NormedField 𝕜
      H : Exists fun y => And (LT.lt (Norm.norm y) 1) (Ne y 0)
      ⊢ ∀ (a : 𝕜), And (LT.lt (Norm.norm a) 1) (Ne a 0) → And (Ne a 0) (Ne (Norm.nor …
    -/
    simp (config := {contextual := true}) [add_comm, ne_of_lt]
    /-
      🎉 no goals
    -/


lemma discreteTopology_of_bddAbove_range_norm {𝕜 : Type*} [NormedField 𝕜]
    (h : BddAbove (Set.range fun k : 𝕜 ↦ ‖k‖)) :
    DiscreteTopology 𝕜 := by
  /-
    𝕜 : Type u_4
    inst✝ : NormedField 𝕜
    h : BddAbove (Set.range fun k => Norm.norm k)
    ⊢ DiscreteTopology 𝕜
  -/
  refine (NormedField.discreteTopology_or_nontriviallyNormedField _).resolve_right ?_
  /-
    𝕜 : Type u_4
    inst✝ : NormedField 𝕜
    h : BddAbove (Set.range fun k => Norm.norm k)
    ⊢ Not (Nonempty (Subtype fun h' => Eq NontriviallyNormedField.toNormedField in …
  -/
  rintro ⟨_, rfl⟩
  /-
    case intro.mk
    𝕜 : Type u_4
    val✝ : NontriviallyNormedField 𝕜
    h : BddAbove (Set.range fun k => Norm.norm k)
    ⊢ False
  -/
  obtain ⟨x, h⟩ := h
  /-
    case intro.mk.intro
    𝕜 : Type u_4
    val✝ : NontriviallyNormedField 𝕜
    x : Real
    h : Membership.mem (upperBounds (Set.range fun k => Norm.norm k)) x
    ⊢ False
  -/
  obtain ⟨k, hk⟩ := NormedField.exists_lt_norm 𝕜 x
  /-
    case intro.mk.intro.intro
    𝕜 : Type u_4
    val✝ : NontriviallyNormedField 𝕜
    x : Real
    h : Membership.mem (upperBounds (Set.range fun k => Norm.norm k)) x
    k : 𝕜
    hk : LT.lt x (Norm.norm k)
    ⊢ False
  -/
  exact hk.not_le (h (Set.mem_range_self k))
  /-
    🎉 no goals
  -/


theorem denseRange_nnnorm : DenseRange (nnnorm : α → ℝ≥0) :=
  dense_of_exists_between fun _ _ hr =>
    let ⟨x, h⟩ := exists_lt_nnnorm_lt α hr
    ⟨‖x‖₊, ⟨x, rfl⟩, h⟩


@[simp]
protected lemma continuousAt_zpow : ContinuousAt (fun x ↦ x ^ n) x ↔ x ≠ 0 ∨ 0 ≤ n := by
  /-
    𝕜 : Type u_4
    inst✝ : NontriviallyNormedField 𝕜
    n : Int
    x : 𝕜
    ⊢ Iff (ContinuousAt (fun x => HPow.hPow x n) x) (Or (Ne x 0) (LE.le 0 n))
  -/
  refine ⟨?_, continuousAt_zpow₀ _ _⟩
  /-
    𝕜 : Type u_4
    inst✝ : NontriviallyNormedField 𝕜
    n : Int
    x : 𝕜
    ⊢ ContinuousAt (fun x => HPow.hPow x n) x → Or (Ne x 0) (LE.le 0 n)
  -/
  contrapose!
  /-
    𝕜 : Type u_4
    inst✝ : NontriviallyNormedField 𝕜
    n : Int
    x : 𝕜
    ⊢ And (Eq x 0) (LT.lt n 0) → Not (ContinuousAt (fun x => HPow.hPow x n) x)
  -/
  rintro ⟨rfl, hm⟩ hc
  exact not_tendsto_atTop_of_tendsto_nhds (hc.tendsto.mono_left nhdsWithin_le_nhds).norm
    (NormedField.tendsto_norm_zpow_nhdsNE_zero_atTop hm)


@[simp]
protected lemma continuousAt_inv : ContinuousAt Inv.inv x ↔ x ≠ 0 := by
  /-
    𝕜 : Type u_4
    inst✝ : NontriviallyNormedField 𝕜
    x : 𝕜
    ⊢ Iff (ContinuousAt Inv.inv x) (Ne x 0)
  -/
  simpa using NormedField.continuousAt_zpow (n := -1) (x := x)
  /-
    🎉 no goals
  -/


lemma lipschitzWith_sub : LipschitzWith 2 (fun (p : ℝ≥0 × ℝ≥0) ↦ p.1 - p.2) := by
  /-
    ⊢ LipschitzWith 2 fun p => HSub.hSub p.1 p.2
  -/
  rw [← isometry_subtype_coe.lipschitzWith_iff]
  have : Isometry (Prod.map ((↑) : ℝ≥0 → ℝ) ((↑) : ℝ≥0 → ℝ)) :=
    isometry_subtype_coe.prod_map isometry_subtype_coe
  convert (((LipschitzWith.prod_fst.comp this.lipschitz).sub
    (LipschitzWith.prod_snd.comp this.lipschitz)).max_const 0)
  /-
    case h.e'_5
    this : Isometry (Prod.map NNReal.toReal NNReal.toReal)
    ⊢ Eq 2 (HAdd.hAdd (HMul.hMul 1 1) (HMul.hMul 1 1))
  -/
  norm_num
  /-
    🎉 no goals
  -/


instance Int.instNormedCommRing : NormedCommRing ℤ where
  __ := instCommRing
  __ := instNormedAddCommGroup
                     /-
                       α : Type u_1
                       β : Type u_2
                       ι : Type u_3
                       m n : Int
                       ⊢ LE.le (Norm.norm (HMul.hMul m n)) (HMul.hMul (Norm.norm m) (Norm.norm n))
                     -/
  norm_mul m n := by simp only [norm, Int.cast_mul, abs_mul, le_rfl]
                     /-
                       🎉 no goals
                     -/


instance Int.instNormOneClass : NormOneClass ℤ :=
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        ⊢ Eq (Norm.norm 1) 1
      -/
  ⟨by simp [← Int.norm_cast_real]⟩
      /-
        🎉 no goals
      -/


instance Rat.instNormedField : NormedField ℚ where
  __ := instField
  __ := instNormedAddCommGroup
                      /-
                        α : Type u_1
                        β : Type u_2
                        ι : Type u_3
                        a b : Rat
                        ⊢ Eq (Norm.norm (HMul.hMul a b)) (HMul.hMul (Norm.norm a) (Norm.norm b))
                      -/
  norm_mul' a b := by simp only [norm, Rat.cast_mul, abs_mul]
                      /-
                        🎉 no goals
                      -/


instance Rat.instDenselyNormedField : DenselyNormedField ℚ where
  lt_norm_lt r₁ r₂ h₀ hr :=
    let ⟨q, h⟩ := exists_rat_btwn hr
           /-
             α : Type u_1
             β : Type u_2
             ι : Type u_3
             r₁ r₂ : Real
             h₀ : LE.le 0 r₁
             hr : LT.lt r₁ r₂
             q : Rat
             h : And (LT.lt r₁ ↑q) (LT.lt (↑q) r₂)
             ⊢ And (LT.lt r₁ (Norm.norm q)) (LT.lt (Norm.norm q) r₂)
           -/
    ⟨q, by rwa [← Rat.norm_cast_real, Real.norm_eq_abs, abs_of_pos (h₀.trans_lt h.1)]⟩
           /-
             🎉 no goals
           -/


lemma NormedField.completeSpace_iff_isComplete_closedBall {K : Type*} [NormedField K] :
    CompleteSpace K ↔ IsComplete (Metric.closedBall 0 1 : Set K) := by
  /-
    K : Type u_4
    inst✝ : NormedField K
    ⊢ Iff (CompleteSpace K) (IsComplete (Metric.closedBall 0 1))
  -/
  constructor <;> intro h
    /-
      case mp
      K : Type u_4
      inst✝ : NormedField K
      h : CompleteSpace K
      ⊢ IsComplete (Metric.closedBall 0 1)
    -/
  · exact Metric.isClosed_ball.isComplete
    /-
      🎉 no goals
    -/
  /-
    case mpr
    K : Type u_4
    inst✝ : NormedField K
    h : IsComplete (Metric.closedBall 0 1)
    ⊢ CompleteSpace K
  -/
  rcases NormedField.discreteTopology_or_nontriviallyNormedField K with _|⟨_, rfl⟩
  · rwa [completeSpace_iff_isComplete_univ,
         ← NormedDivisionRing.unitClosedBall_eq_univ_of_discrete]
  /-
    case mpr.inr.intro.mk
    K : Type u_4
    val✝ : NontriviallyNormedField K
    h : IsComplete (Metric.closedBall 0 1)
    ⊢ CompleteSpace K
  -/
  refine Metric.complete_of_cauchySeq_tendsto fun u hu ↦ ?_
  /-
    case mpr.inr.intro.mk
    K : Type u_4
    val✝ : NontriviallyNormedField K
    h : IsComplete (Metric.closedBall 0 1)
    u : Nat → K
    hu : CauchySeq u
    ⊢ Exists fun a => Filter.Tendsto u Filter.atTop (nhds a)
  -/
  obtain ⟨k, hk⟩ := hu.norm_bddAbove
  /-
    case mpr.inr.intro.mk.intro
    K : Type u_4
    val✝ : NontriviallyNormedField K
    h : IsComplete (Metric.closedBall 0 1)
    u : Nat → K
    hu : CauchySeq u
    k : Real
    hk : Membership.mem (upperBounds (Set.range fun n => Norm.norm (u n))) k
    ⊢ Exists fun a => Filter.Tendsto u Filter.atTop (nhds a)
  -/
  have kpos : 0 ≤ k := (_root_.norm_nonneg (u 0)).trans (hk (by simp))
  /-
    case mpr.inr.intro.mk.intro
    K : Type u_4
    val✝ : NontriviallyNormedField K
    h : IsComplete (Metric.closedBall 0 1)
    u : Nat → K
    hu : CauchySeq u
    k : Real
    hk : Membership.mem (upperBounds (Set.range fun n => Norm.norm (u n))) k
    kpos : LE.le 0 k
    ⊢ Exists fun a => Filter.Tendsto u Filter.atTop (nhds a)
  -/
  obtain ⟨x, hx⟩ := NormedField.exists_lt_norm K k
  /-
    case mpr.inr.intro.mk.intro.intro
    K : Type u_4
    val✝ : NontriviallyNormedField K
    h : IsComplete (Metric.closedBall 0 1)
    u : Nat → K
    hu : CauchySeq u
    k : Real
    hk : Membership.mem (upperBounds (Set.range fun n => Norm.norm (u n))) k
    kpos : LE.le 0 k
    x : K
    hx : LT.lt k (Norm.norm x)
    ⊢ Exists fun a => Filter.Tendsto u Filter.atTop (nhds a)
  -/
  have hu' : CauchySeq ((· / x) ∘ u) := (uniformContinuous_div_const' x).comp_cauchySeq hu
  have hb : ∀ n, ((· / x) ∘ u) n ∈ Metric.closedBall 0 1 := by
    intro
    simp only [Function.comp_apply, Metric.mem_closedBall, dist_zero_right, norm_div]
    rw [div_le_one (kpos.trans_lt hx)]
    exact hx.le.trans' (hk (by simp))
  /-
    case mpr.inr.intro.mk.intro.intro
    K : Type u_4
    val✝ : NontriviallyNormedField K
    h : IsComplete (Metric.closedBall 0 1)
    u : Nat → K
    hu : CauchySeq u
    k : Real
    hk : Membership.mem (upperBounds (Set.range fun n => Norm.norm (u n))) k
    kpos : LE.le 0 k
    x : K
    hx : LT.lt k (Norm.norm x)
    hu' : CauchySeq (Function.comp (fun x_1 => HDiv.hDiv x_1 x) u)
    hb : ∀ (n : Nat), Membership.mem (Metric.closedBall 0 1) (Function.comp (fun x …
    ⊢ Exists fun a => Filter.Tendsto u Filter.atTop (nhds a)
  -/
  obtain ⟨a, -, ha'⟩ := cauchySeq_tendsto_of_isComplete h hb hu'
  /-
    case mpr.inr.intro.mk.intro.intro.intro.intro
    K : Type u_4
    val✝ : NontriviallyNormedField K
    h : IsComplete (Metric.closedBall 0 1)
    u : Nat → K
    hu : CauchySeq u
    k : Real
    hk : Membership.mem (upperBounds (Set.range fun n => Norm.norm (u n))) k
    kpos : LE.le 0 k
    x : K
    hx : LT.lt k (Norm.norm x)
    hu' : CauchySeq (Function.comp (fun x_1 => HDiv.hDiv x_1 x) u)
    hb : ∀ (n : Nat), Membership.mem (Metric.closedBall 0 1) (Function.comp (fun x …
    a : K
    ha' : Filter.Tendsto (Function.comp (fun x_1 => HDiv.hDiv x_1 x) u) Filter.atT …
    ⊢ Exists fun a => Filter.Tendsto u Filter.atTop (nhds a)
  -/
  refine ⟨a * x, (((continuous_mul_right x).tendsto a).comp ha').congr ?_⟩
  have hx' : x ≠ 0 := by
    contrapose! hx
    simp [hx, kpos]
  /-
    case mpr.inr.intro.mk.intro.intro.intro.intro
    K : Type u_4
    val✝ : NontriviallyNormedField K
    h : IsComplete (Metric.closedBall 0 1)
    u : Nat → K
    hu : CauchySeq u
    k : Real
    hk : Membership.mem (upperBounds (Set.range fun n => Norm.norm (u n))) k
    kpos : LE.le 0 k
    x : K
    hx : LT.lt k (Norm.norm x)
    hu' : CauchySeq (Function.comp (fun x_1 => HDiv.hDiv x_1 x) u)
    hb : ∀ (n : Nat), Membership.mem (Metric.closedBall 0 1) (Function.comp (fun x …
    a : K
    ha' : Filter.Tendsto (Function.comp (fun x_1 => HDiv.hDiv x_1 x) u) Filter.atT …
    hx' : Ne x 0
    ⊢ ∀ (x_1 : Nat), Eq (Function.comp (fun b => HMul.hMul b x) (Function.comp (fu …
  -/
  simp [div_mul_cancel₀ _ hx']
  /-
    🎉 no goals
  -/



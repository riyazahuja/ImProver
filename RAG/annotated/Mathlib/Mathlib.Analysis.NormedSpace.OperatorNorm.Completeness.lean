/-- Construct a bundled continuous (semi)linear map from a map `f : E → F` and a proof of the fact
that it belongs to the closure of the image of a bounded set `s : Set (E →SL[σ₁₂] F)` under coercion
to function. Coercion to function of the result is definitionally equal to `f`. -/
@[simps! (config := .asFn) apply]
def ofMemClosureImageCoeBounded (f : E' → F) {s : Set (E' →SL[σ₁₂] F)} (hs : IsBounded s)
    (hf : f ∈ closure (((↑) : (E' →SL[σ₁₂] F) → E' → F) '' s)) : E' →SL[σ₁₂] F := by
  -- `f` is a linear map due to `linearMapOfMemClosureRangeCoe`
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_3
    F : Type u_4
    Fₗ : Type u_5
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedAddCommGroup Fₗ
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NontriviallyNormedField 𝕜₂
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedSpace 𝕜₂ F
    inst✝³ : NormedSpace 𝕜 Fₗ
    σ₁₂ : RingHom 𝕜 𝕜₂
    f✝ g : ContinuousLinearMap σ₁₂ E F
    E' : Type u_6
    inst✝² : SeminormedAddCommGroup E'
    inst✝¹ : NormedSpace 𝕜 E'
    inst✝ : RingHomIsometric σ₁₂
    f : E' → F
    s : Set (ContinuousLinearMap σ₁₂ E' F)
    hs : Bornology.IsBounded s
    hf : Membership.mem (closure (Set.image DFunLike.coe s)) f
    ⊢ ContinuousLinearMap σ₁₂ E' F
  -/
  refine (linearMapOfMemClosureRangeCoe f ?_).mkContinuousOfExistsBound ?_
    /-
      case refine_1
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_3
      F : Type u_4
      Fₗ : Type u_5
      inst✝¹⁰ : NormedAddCommGroup E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedAddCommGroup Fₗ
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NontriviallyNormedField 𝕜₂
      inst✝⁵ : NormedSpace 𝕜 E
      inst✝⁴ : NormedSpace 𝕜₂ F
      inst✝³ : NormedSpace 𝕜 Fₗ
      σ₁₂ : RingHom 𝕜 𝕜₂
      f✝ g : ContinuousLinearMap σ₁₂ E F
      E' : Type u_6
      inst✝² : SeminormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      inst✝ : RingHomIsometric σ₁₂
      f : E' → F
      s : Set (ContinuousLinearMap σ₁₂ E' F)
      hs : Bornology.IsBounded s
      hf : Membership.mem (closure (Set.image DFunLike.coe s)) f
      ⊢ Membership.mem (closure (Set.range DFunLike.coe)) f
    -/
  · refine closure_mono (image_subset_iff.2 fun g _ => ?_) hf
    /-
      case refine_1
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_3
      F : Type u_4
      Fₗ : Type u_5
      inst✝¹⁰ : NormedAddCommGroup E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedAddCommGroup Fₗ
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NontriviallyNormedField 𝕜₂
      inst✝⁵ : NormedSpace 𝕜 E
      inst✝⁴ : NormedSpace 𝕜₂ F
      inst✝³ : NormedSpace 𝕜 Fₗ
      σ₁₂ : RingHom 𝕜 𝕜₂
      f✝ g✝ : ContinuousLinearMap σ₁₂ E F
      E' : Type u_6
      inst✝² : SeminormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      inst✝ : RingHomIsometric σ₁₂
      f : E' → F
      s : Set (ContinuousLinearMap σ₁₂ E' F)
      hs : Bornology.IsBounded s
      hf : Membership.mem (closure (Set.image DFunLike.coe s)) f
      g : ContinuousLinearMap σ₁₂ E' F
      x✝ : Membership.mem s g
      ⊢ Membership.mem (Set.preimage DFunLike.coe (Set.range DFunLike.coe)) g
    -/
    exact ⟨g, rfl⟩
    /-
      🎉 no goals
    -/
  · -- We need to show that `f` has bounded norm. Choose `C` such that `‖g‖ ≤ C` for all `g ∈ s`.
    /-
      case refine_2
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_3
      F : Type u_4
      Fₗ : Type u_5
      inst✝¹⁰ : NormedAddCommGroup E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedAddCommGroup Fₗ
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NontriviallyNormedField 𝕜₂
      inst✝⁵ : NormedSpace 𝕜 E
      inst✝⁴ : NormedSpace 𝕜₂ F
      inst✝³ : NormedSpace 𝕜 Fₗ
      σ₁₂ : RingHom 𝕜 𝕜₂
      f✝ g : ContinuousLinearMap σ₁₂ E F
      E' : Type u_6
      inst✝² : SeminormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      inst✝ : RingHomIsometric σ₁₂
      f : E' → F
      s : Set (ContinuousLinearMap σ₁₂ E' F)
      hs : Bornology.IsBounded s
      hf : Membership.mem (closure (Set.image DFunLike.coe s)) f
      ⊢ Exists fun C => ∀ (x : E'), LE.le (Norm.norm ((linearMapOfMemClosureRangeCoe …
    -/
    rcases isBounded_iff_forall_norm_le.1 hs with ⟨C, hC⟩
    -- Then `‖g x‖ ≤ C * ‖x‖` for all `g ∈ s`, `x : E`, hence `‖f x‖ ≤ C * ‖x‖` for all `x`.
    have : ∀ x, IsClosed { g : E' → F | ‖g x‖ ≤ C * ‖x‖ } := fun x =>
      isClosed_Iic.preimage (@continuous_apply E' (fun _ => F) _ x).norm
    /-
      case refine_2.intro
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_3
      F : Type u_4
      Fₗ : Type u_5
      inst✝¹⁰ : NormedAddCommGroup E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedAddCommGroup Fₗ
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NontriviallyNormedField 𝕜₂
      inst✝⁵ : NormedSpace 𝕜 E
      inst✝⁴ : NormedSpace 𝕜₂ F
      inst✝³ : NormedSpace 𝕜 Fₗ
      σ₁₂ : RingHom 𝕜 𝕜₂
      f✝ g : ContinuousLinearMap σ₁₂ E F
      E' : Type u_6
      inst✝² : SeminormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      inst✝ : RingHomIsometric σ₁₂
      f : E' → F
      s : Set (ContinuousLinearMap σ₁₂ E' F)
      hs : Bornology.IsBounded s
      hf : Membership.mem (closure (Set.image DFunLike.coe s)) f
      C : Real
      hC : ∀ (x : ContinuousLinearMap σ₁₂ E' F), Membership.mem s x → LE.le (Norm.no …
      this : ∀ (x : E'), IsClosed (setOf fun g => LE.le (Norm.norm (g x)) (HMul.hMul …
      ⊢ Exists fun C => ∀ (x : E'), LE.le (Norm.norm ((linearMapOfMemClosureRangeCoe …
    -/
    refine ⟨C, fun x => (this x).closure_subset_iff.2 (image_subset_iff.2 fun g hg => ?_) hf⟩
    /-
      case refine_2.intro
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_3
      F : Type u_4
      Fₗ : Type u_5
      inst✝¹⁰ : NormedAddCommGroup E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedAddCommGroup Fₗ
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NontriviallyNormedField 𝕜₂
      inst✝⁵ : NormedSpace 𝕜 E
      inst✝⁴ : NormedSpace 𝕜₂ F
      inst✝³ : NormedSpace 𝕜 Fₗ
      σ₁₂ : RingHom 𝕜 𝕜₂
      f✝ g✝ : ContinuousLinearMap σ₁₂ E F
      E' : Type u_6
      inst✝² : SeminormedAddCommGroup E'
      inst✝¹ : NormedSpace 𝕜 E'
      inst✝ : RingHomIsometric σ₁₂
      f : E' → F
      s : Set (ContinuousLinearMap σ₁₂ E' F)
      hs : Bornology.IsBounded s
      hf : Membership.mem (closure (Set.image DFunLike.coe s)) f
      C : Real
      hC : ∀ (x : ContinuousLinearMap σ₁₂ E' F), Membership.mem s x → LE.le (Norm.no …
      this : ∀ (x : E'), IsClosed (setOf fun g => LE.le (Norm.norm (g x)) (HMul.hMul …
      x : E'
      g : ContinuousLinearMap σ₁₂ E' F
      hg : Membership.mem s g
      ⊢ Membership.mem (Set.preimage DFunLike.coe (setOf fun g => LE.le (Norm.norm ( …
    -/
    exact g.le_of_opNorm_le (hC _ hg) _
    /-
      🎉 no goals
    -/


/-- Let `f : E → F` be a map, let `g : α → E →SL[σ₁₂] F` be a family of continuous (semi)linear maps
that takes values in a bounded set and converges to `f` pointwise along a nontrivial filter. Then
`f` is a continuous (semi)linear map. -/
@[simps! (config := .asFn) apply]
def ofTendstoOfBoundedRange {α : Type*} {l : Filter α} [l.NeBot] (f : E' → F)
    (g : α → E' →SL[σ₁₂] F) (hf : Tendsto (fun a x => g a x) l (𝓝 f))
    (hg : IsBounded (Set.range g)) : E' →SL[σ₁₂] F :=
  ofMemClosureImageCoeBounded f hg <| mem_closure_of_tendsto hf <|
    Eventually.of_forall fun _ => mem_image_of_mem _ <| Set.mem_range_self _


/-- If a Cauchy sequence of continuous linear map converges to a continuous linear map pointwise,
then it converges to the same map in norm. This lemma is used to prove that the space of continuous
linear maps is complete provided that the codomain is a complete space. -/
theorem tendsto_of_tendsto_pointwise_of_cauchySeq {f : ℕ → E' →SL[σ₁₂] F} {g : E' →SL[σ₁₂] F}
    (hg : Tendsto (fun n x => f n x) atTop (𝓝 g)) (hf : CauchySeq f) : Tendsto f atTop (𝓝 g) := by
  /- Since `f` is a Cauchy sequence, there exists `b → 0` such that `‖f n - f m‖ ≤ b N` for any
    `m, n ≥ N`. -/
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    E' : Type u_6
    inst✝² : SeminormedAddCommGroup E'
    inst✝¹ : NormedSpace 𝕜 E'
    inst✝ : RingHomIsometric σ₁₂
    f : Nat → ContinuousLinearMap σ₁₂ E' F
    g : ContinuousLinearMap σ₁₂ E' F
    hg : Filter.Tendsto (fun n x => (f n) x) Filter.atTop (nhds ⇑g)
    hf : CauchySeq f
    ⊢ Filter.Tendsto f Filter.atTop (nhds g)
  -/
  rcases cauchySeq_iff_le_tendsto_0.1 hf with ⟨b, hb₀, hfb, hb_lim⟩
  -- Since `b → 0`, it suffices to show that `‖f n x - g x‖ ≤ b n * ‖x‖` for all `n` and `x`.
  suffices ∀ n x, ‖f n x - g x‖ ≤ b n * ‖x‖ from
    tendsto_iff_norm_sub_tendsto_zero.2
    (squeeze_zero (fun n => norm_nonneg _) (fun n => opNorm_le_bound _ (hb₀ n) (this n)) hb_lim)
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    E' : Type u_6
    inst✝² : SeminormedAddCommGroup E'
    inst✝¹ : NormedSpace 𝕜 E'
    inst✝ : RingHomIsometric σ₁₂
    f : Nat → ContinuousLinearMap σ₁₂ E' F
    g : ContinuousLinearMap σ₁₂ E' F
    hg : Filter.Tendsto (fun n x => (f n) x) Filter.atTop (nhds ⇑g)
    hf : CauchySeq f
    b : Nat → Real
    hb₀ : ∀ (n : Nat), LE.le 0 (b n)
    hfb : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (f n) (f m)) ( …
    hb_lim : Filter.Tendsto b Filter.atTop (nhds 0)
    ⊢ ∀ (n : Nat) (x : E'), LE.le (Norm.norm (HSub.hSub ((f n) x) (g x))) (HMul.hM …
  -/
  intro n x
  -- Note that `f m x → g x`, hence `‖f n x - f m x‖ → ‖f n x - g x‖` as `m → ∞`
  have : Tendsto (fun m => ‖f n x - f m x‖) atTop (𝓝 ‖f n x - g x‖) :=
    (tendsto_const_nhds.sub <| tendsto_pi_nhds.1 hg _).norm
  -- Thus it suffices to verify `‖f n x - f m x‖ ≤ b n * ‖x‖` for `m ≥ n`.
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    E' : Type u_6
    inst✝² : SeminormedAddCommGroup E'
    inst✝¹ : NormedSpace 𝕜 E'
    inst✝ : RingHomIsometric σ₁₂
    f : Nat → ContinuousLinearMap σ₁₂ E' F
    g : ContinuousLinearMap σ₁₂ E' F
    hg : Filter.Tendsto (fun n x => (f n) x) Filter.atTop (nhds ⇑g)
    hf : CauchySeq f
    b : Nat → Real
    hb₀ : ∀ (n : Nat), LE.le 0 (b n)
    hfb : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (f n) (f m)) ( …
    hb_lim : Filter.Tendsto b Filter.atTop (nhds 0)
    n : Nat
    x : E'
    this : Filter.Tendsto (fun m => Norm.norm (HSub.hSub ((f n) x) ((f m) x))) Fil …
    ⊢ LE.le (Norm.norm (HSub.hSub ((f n) x) (g x))) (HMul.hMul (b n) (Norm.norm x))
  -/
  refine le_of_tendsto this (eventually_atTop.2 ⟨n, fun m hm => ?_⟩)
  -- This inequality follows from `‖f n - f m‖ ≤ b n`.
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    E' : Type u_6
    inst✝² : SeminormedAddCommGroup E'
    inst✝¹ : NormedSpace 𝕜 E'
    inst✝ : RingHomIsometric σ₁₂
    f : Nat → ContinuousLinearMap σ₁₂ E' F
    g : ContinuousLinearMap σ₁₂ E' F
    hg : Filter.Tendsto (fun n x => (f n) x) Filter.atTop (nhds ⇑g)
    hf : CauchySeq f
    b : Nat → Real
    hb₀ : ∀ (n : Nat), LE.le 0 (b n)
    hfb : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (f n) (f m)) ( …
    hb_lim : Filter.Tendsto b Filter.atTop (nhds 0)
    n : Nat
    x : E'
    this : Filter.Tendsto (fun m => Norm.norm (HSub.hSub ((f n) x) ((f m) x))) Fil …
    m : Nat
    hm : GE.ge m n
    ⊢ LE.le (Norm.norm (HSub.hSub ((f n) x) ((f m) x))) (HMul.hMul (b n) (Norm.nor …
  -/
  exact (f n - f m).le_of_opNorm_le (hfb _ _ _ le_rfl hm) _
  /-
    🎉 no goals
  -/


/-- If the target space is complete, the space of continuous linear maps with its norm is also
complete. This works also if the source space is seminormed. -/
instance [CompleteSpace F] : CompleteSpace (E' →SL[σ₁₂] F) := by
  -- We show that every Cauchy sequence converges.
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_3
    F : Type u_4
    Fₗ : Type u_5
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedAddCommGroup Fₗ
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NontriviallyNormedField 𝕜₂
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜₂ F
    inst✝⁴ : NormedSpace 𝕜 Fₗ
    σ₁₂ : RingHom 𝕜 𝕜₂
    f g : ContinuousLinearMap σ₁₂ E F
    E' : Type u_6
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedSpace 𝕜 E'
    inst✝¹ : RingHomIsometric σ₁₂
    inst✝ : CompleteSpace F
    ⊢ CompleteSpace (ContinuousLinearMap σ₁₂ E' F)
  -/
  refine Metric.complete_of_cauchySeq_tendsto fun f hf => ?_
  -- The evaluation at any point `v : E` is Cauchy.
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_3
    F : Type u_4
    Fₗ : Type u_5
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedAddCommGroup Fₗ
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NontriviallyNormedField 𝕜₂
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜₂ F
    inst✝⁴ : NormedSpace 𝕜 Fₗ
    σ₁₂ : RingHom 𝕜 𝕜₂
    f✝ g : ContinuousLinearMap σ₁₂ E F
    E' : Type u_6
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedSpace 𝕜 E'
    inst✝¹ : RingHomIsometric σ₁₂
    inst✝ : CompleteSpace F
    f : Nat → ContinuousLinearMap σ₁₂ E' F
    hf : CauchySeq f
    ⊢ Exists fun a => Filter.Tendsto f Filter.atTop (nhds a)
  -/
  have cau : ∀ v, CauchySeq fun n => f n v := fun v => hf.map (lipschitz_apply v).uniformContinuous
  -- We assemble the limits points of those Cauchy sequences
  -- (which exist as `F` is complete)
  -- into a function which we call `G`.
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_3
    F : Type u_4
    Fₗ : Type u_5
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedAddCommGroup Fₗ
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NontriviallyNormedField 𝕜₂
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜₂ F
    inst✝⁴ : NormedSpace 𝕜 Fₗ
    σ₁₂ : RingHom 𝕜 𝕜₂
    f✝ g : ContinuousLinearMap σ₁₂ E F
    E' : Type u_6
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedSpace 𝕜 E'
    inst✝¹ : RingHomIsometric σ₁₂
    inst✝ : CompleteSpace F
    f : Nat → ContinuousLinearMap σ₁₂ E' F
    hf : CauchySeq f
    cau : ∀ (v : E'), CauchySeq fun n => (f n) v
    ⊢ Exists fun a => Filter.Tendsto f Filter.atTop (nhds a)
  -/
  choose G hG using fun v => cauchySeq_tendsto_of_complete (cau v)
  -- Next, we show that this `G` is a continuous linear map.
  -- This is done in `ContinuousLinearMap.ofTendstoOfBoundedRange`.
  set Glin : E' →SL[σ₁₂] F :=
    ofTendstoOfBoundedRange _ _ (tendsto_pi_nhds.mpr hG) hf.isBounded_range
  -- Finally, `f n` converges to `Glin` in norm because of
  -- `ContinuousLinearMap.tendsto_of_tendsto_pointwise_of_cauchySeq`
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_3
    F : Type u_4
    Fₗ : Type u_5
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedAddCommGroup Fₗ
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : NontriviallyNormedField 𝕜₂
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedSpace 𝕜₂ F
    inst✝⁴ : NormedSpace 𝕜 Fₗ
    σ₁₂ : RingHom 𝕜 𝕜₂
    f✝ g : ContinuousLinearMap σ₁₂ E F
    E' : Type u_6
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedSpace 𝕜 E'
    inst✝¹ : RingHomIsometric σ₁₂
    inst✝ : CompleteSpace F
    f : Nat → ContinuousLinearMap σ₁₂ E' F
    hf : CauchySeq f
    cau : ∀ (v : E'), CauchySeq fun n => (f n) v
    G : E' → F
    hG : ∀ (v : E'), Filter.Tendsto (fun n => (f n) v) Filter.atTop (nhds (G v))
    Glin : ContinuousLinearMap σ₁₂ E' F := ContinuousLinearMap.ofTendstoOfBoundedR …
    ⊢ Exists fun a => Filter.Tendsto f Filter.atTop (nhds a)
  -/
  exact ⟨Glin, tendsto_of_tendsto_pointwise_of_cauchySeq (tendsto_pi_nhds.2 hG) hf⟩
  /-
    🎉 no goals
  -/


/-- Let `s` be a bounded set in the space of continuous (semi)linear maps `E →SL[σ] F` taking values
in a proper space. Then `s` interpreted as a set in the space of maps `E → F` with topology of
pointwise convergence is precompact: its closure is a compact set. -/
theorem isCompact_closure_image_coe_of_bounded [ProperSpace F] {s : Set (E' →SL[σ₁₂] F)}
    (hb : IsBounded s) : IsCompact (closure (((↑) : (E' →SL[σ₁₂] F) → E' → F) '' s)) :=
  have : ∀ x, IsCompact (closure (apply' F σ₁₂ x '' s)) := fun x =>
    ((apply' F σ₁₂ x).lipschitz.isBounded_image hb).isCompact_closure
  (isCompact_pi_infinite this).closure_of_subset
    (image_subset_iff.2 fun _ hg _ => subset_closure <| mem_image_of_mem _ hg)


/-- Let `s` be a bounded set in the space of continuous (semi)linear maps `E →SL[σ] F` taking values
in a proper space. If `s` interpreted as a set in the space of maps `E → F` with topology of
pointwise convergence is closed, then it is compact.

TODO: reformulate this in terms of a type synonym with the right topology. -/
theorem isCompact_image_coe_of_bounded_of_closed_image [ProperSpace F] {s : Set (E' →SL[σ₁₂] F)}
    (hb : IsBounded s) (hc : IsClosed (((↑) : (E' →SL[σ₁₂] F) → E' → F) '' s)) :
    IsCompact (((↑) : (E' →SL[σ₁₂] F) → E' → F) '' s) :=
  hc.closure_eq ▸ isCompact_closure_image_coe_of_bounded hb


/-- If a set `s` of semilinear functions is bounded and is closed in the weak-* topology, then its
image under coercion to functions `E → F` is a closed set. We don't have a name for `E →SL[σ] F`
with weak-* topology in `mathlib`, so we use an equivalent condition (see `isClosed_induced_iff'`).

TODO: reformulate this in terms of a type synonym with the right topology. -/
theorem isClosed_image_coe_of_bounded_of_weak_closed {s : Set (E' →SL[σ₁₂] F)} (hb : IsBounded s)
    (hc : ∀ f : E' →SL[σ₁₂] F,
      (⇑f : E' → F) ∈ closure (((↑) : (E' →SL[σ₁₂] F) → E' → F) '' s) → f ∈ s) :
    IsClosed (((↑) : (E' →SL[σ₁₂] F) → E' → F) '' s) :=
  isClosed_of_closure_subset fun f hf =>
    ⟨ofMemClosureImageCoeBounded f hb hf, hc (ofMemClosureImageCoeBounded f hb hf) hf, rfl⟩


/-- If a set `s` of semilinear functions is bounded and is closed in the weak-* topology, then its
image under coercion to functions `E → F` is a compact set. We don't have a name for `E →SL[σ] F`
with weak-* topology in `mathlib`, so we use an equivalent condition (see `isClosed_induced_iff'`).
-/
theorem isCompact_image_coe_of_bounded_of_weak_closed [ProperSpace F] {s : Set (E' →SL[σ₁₂] F)}
    (hb : IsBounded s) (hc : ∀ f : E' →SL[σ₁₂] F,
      (⇑f : E' → F) ∈ closure (((↑) : (E' →SL[σ₁₂] F) → E' → F) '' s) → f ∈ s) :
    IsCompact (((↑) : (E' →SL[σ₁₂] F) → E' → F) '' s) :=
  isCompact_image_coe_of_bounded_of_closed_image hb <|
    isClosed_image_coe_of_bounded_of_weak_closed hb hc


/-- A closed ball is closed in the weak-* topology. We don't have a name for `E →SL[σ] F` with
weak-* topology in `mathlib`, so we use an equivalent condition (see `isClosed_induced_iff'`). -/
theorem is_weak_closed_closedBall (f₀ : E' →SL[σ₁₂] F) (r : ℝ) ⦃f : E' →SL[σ₁₂] F⦄
    (hf : ⇑f ∈ closure (((↑) : (E' →SL[σ₁₂] F) → E' → F) '' closedBall f₀ r)) :
    f ∈ closedBall f₀ r := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    E' : Type u_6
    inst✝² : SeminormedAddCommGroup E'
    inst✝¹ : NormedSpace 𝕜 E'
    inst✝ : RingHomIsometric σ₁₂
    f₀ : ContinuousLinearMap σ₁₂ E' F
    r : Real
    f : ContinuousLinearMap σ₁₂ E' F
    hf : Membership.mem (closure (Set.image DFunLike.coe (Metric.closedBall f₀ r)) …
    ⊢ Membership.mem (Metric.closedBall f₀ r) f
  -/
  have hr : 0 ≤ r := nonempty_closedBall.1 (closure_nonempty_iff.1 ⟨_, hf⟩).of_image
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    E' : Type u_6
    inst✝² : SeminormedAddCommGroup E'
    inst✝¹ : NormedSpace 𝕜 E'
    inst✝ : RingHomIsometric σ₁₂
    f₀ : ContinuousLinearMap σ₁₂ E' F
    r : Real
    f : ContinuousLinearMap σ₁₂ E' F
    hf : Membership.mem (closure (Set.image DFunLike.coe (Metric.closedBall f₀ r)) …
    hr : LE.le 0 r
    ⊢ Membership.mem (Metric.closedBall f₀ r) f
  -/
  refine mem_closedBall_iff_norm.2 (opNorm_le_bound _ hr fun x => ?_)
  have : IsClosed { g : E' → F | ‖g x - f₀ x‖ ≤ r * ‖x‖ } :=
    isClosed_Iic.preimage ((@continuous_apply E' (fun _ => F) _ x).sub continuous_const).norm
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    E' : Type u_6
    inst✝² : SeminormedAddCommGroup E'
    inst✝¹ : NormedSpace 𝕜 E'
    inst✝ : RingHomIsometric σ₁₂
    f₀ : ContinuousLinearMap σ₁₂ E' F
    r : Real
    f : ContinuousLinearMap σ₁₂ E' F
    hf : Membership.mem (closure (Set.image DFunLike.coe (Metric.closedBall f₀ r)) …
    hr : LE.le 0 r
    x : E'
    this : IsClosed (setOf fun g => LE.le (Norm.norm (HSub.hSub (g x) (f₀ x))) (HM …
    ⊢ LE.le (Norm.norm ((HSub.hSub f f₀) x)) (HMul.hMul r (Norm.norm x))
  -/
  refine this.closure_subset_iff.2 (image_subset_iff.2 fun g hg => ?_) hf
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NontriviallyNormedField 𝕜₂
    inst✝³ : NormedSpace 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    E' : Type u_6
    inst✝² : SeminormedAddCommGroup E'
    inst✝¹ : NormedSpace 𝕜 E'
    inst✝ : RingHomIsometric σ₁₂
    f₀ : ContinuousLinearMap σ₁₂ E' F
    r : Real
    f : ContinuousLinearMap σ₁₂ E' F
    hf : Membership.mem (closure (Set.image DFunLike.coe (Metric.closedBall f₀ r)) …
    hr : LE.le 0 r
    x : E'
    this : IsClosed (setOf fun g => LE.le (Norm.norm (HSub.hSub (g x) (f₀ x))) (HM …
    g : ContinuousLinearMap σ₁₂ E' F
    hg : Membership.mem (Metric.closedBall f₀ r) g
    ⊢ Membership.mem (Set.preimage DFunLike.coe (setOf fun g => LE.le (Norm.norm ( …
  -/
  exact (g - f₀).le_of_opNorm_le (mem_closedBall_iff_norm.1 hg) _
  /-
    🎉 no goals
  -/


/-- The set of functions `f : E → F` that represent continuous linear maps `f : E →SL[σ₁₂] F`
at distance `≤ r` from `f₀ : E →SL[σ₁₂] F` is closed in the topology of pointwise convergence.
This is one of the key steps in the proof of the **Banach-Alaoglu** theorem. -/
theorem isClosed_image_coe_closedBall (f₀ : E →SL[σ₁₂] F) (r : ℝ) :
    IsClosed (((↑) : (E →SL[σ₁₂] F) → E → F) '' closedBall f₀ r) :=
  isClosed_image_coe_of_bounded_of_weak_closed isBounded_closedBall (is_weak_closed_closedBall f₀ r)


/-- **Banach-Alaoglu** theorem. The set of functions `f : E → F` that represent continuous linear
maps `f : E →SL[σ₁₂] F` at distance `≤ r` from `f₀ : E →SL[σ₁₂] F` is compact in the topology of
pointwise convergence. Other versions of this theorem can be found in
`Analysis.Normed.Module.WeakDual`. -/
theorem isCompact_image_coe_closedBall [ProperSpace F] (f₀ : E →SL[σ₁₂] F) (r : ℝ) :
    IsCompact (((↑) : (E →SL[σ₁₂] F) → E → F) '' closedBall f₀ r) :=
  isCompact_image_coe_of_bounded_of_weak_closed isBounded_closedBall <|
    is_weak_closed_closedBall f₀ r


/-- Extension of a continuous linear map `f : E →SL[σ₁₂] F`, with `E` a normed space and `F` a
complete normed space, along a uniform and dense embedding `e : E →L[𝕜] Fₗ`. -/
def extend : Fₗ →SL[σ₁₂] F :=
  -- extension of `f` is continuous
  have cont := (uniformContinuous_uniformly_extend h_e h_dense f.uniformContinuous).continuous
  -- extension of `f` agrees with `f` on the domain of the embedding `e`
  have eq := uniformly_extend_of_ind h_e h_dense f.uniformContinuous
  { toFun := (h_e.isDenseInducing h_dense).extend f
    map_add' := by
      /-
        𝕜 : Type u_1
        𝕜₂ : Type u_2
        E : Type u_3
        F : Type u_4
        Fₗ : Type u_5
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : NormedAddCommGroup F
        inst✝⁶ : NormedAddCommGroup Fₗ
        inst✝⁵ : NontriviallyNormedField 𝕜
        inst✝⁴ : NontriviallyNormedField 𝕜₂
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedSpace 𝕜₂ F
        inst✝¹ : NormedSpace 𝕜 Fₗ
        σ₁₂ : RingHom 𝕜 𝕜₂
        f g : ContinuousLinearMap σ₁₂ E F
        inst✝ : CompleteSpace F
        e : ContinuousLinearMap (RingHom.id 𝕜) E Fₗ
        h_dense : DenseRange ⇑e
        h_e : IsUniformInducing ⇑e
        cont : Continuous (⋯.extend ⇑f)
        eq : ∀ (b : E), Eq (⋯.extend (⇑f) (e b)) (f b)
        ⊢ ∀ (x y : Fₗ), Eq (⋯.extend (⇑f) (HAdd.hAdd x y)) (HAdd.hAdd (⋯.extend (⇑f) x …
      -/
      refine h_dense.induction_on₂ ?_ ?_
      · exact isClosed_eq (cont.comp continuous_add)
          ((cont.comp continuous_fst).add (cont.comp continuous_snd))
        /-
          case refine_2
          𝕜 : Type u_1
          𝕜₂ : Type u_2
          E : Type u_3
          F : Type u_4
          Fₗ : Type u_5
          inst✝⁸ : NormedAddCommGroup E
          inst✝⁷ : NormedAddCommGroup F
          inst✝⁶ : NormedAddCommGroup Fₗ
          inst✝⁵ : NontriviallyNormedField 𝕜
          inst✝⁴ : NontriviallyNormedField 𝕜₂
          inst✝³ : NormedSpace 𝕜 E
          inst✝² : NormedSpace 𝕜₂ F
          inst✝¹ : NormedSpace 𝕜 Fₗ
          σ₁₂ : RingHom 𝕜 𝕜₂
          f g : ContinuousLinearMap σ₁₂ E F
          inst✝ : CompleteSpace F
          e : ContinuousLinearMap (RingHom.id 𝕜) E Fₗ
          h_dense : DenseRange ⇑e
          h_e : IsUniformInducing ⇑e
          cont : Continuous (⋯.extend ⇑f)
          eq : ∀ (b : E), Eq (⋯.extend (⇑f) (e b)) (f b)
          ⊢ ∀ (a₁ a₂ : E), Eq (⋯.extend (⇑f) (HAdd.hAdd (e a₁) (e a₂))) (HAdd.hAdd (⋯.ex …
        -/
      · intro x y
        /-
          case refine_2
          𝕜 : Type u_1
          𝕜₂ : Type u_2
          E : Type u_3
          F : Type u_4
          Fₗ : Type u_5
          inst✝⁸ : NormedAddCommGroup E
          inst✝⁷ : NormedAddCommGroup F
          inst✝⁶ : NormedAddCommGroup Fₗ
          inst✝⁵ : NontriviallyNormedField 𝕜
          inst✝⁴ : NontriviallyNormedField 𝕜₂
          inst✝³ : NormedSpace 𝕜 E
          inst✝² : NormedSpace 𝕜₂ F
          inst✝¹ : NormedSpace 𝕜 Fₗ
          σ₁₂ : RingHom 𝕜 𝕜₂
          f g : ContinuousLinearMap σ₁₂ E F
          inst✝ : CompleteSpace F
          e : ContinuousLinearMap (RingHom.id 𝕜) E Fₗ
          h_dense : DenseRange ⇑e
          h_e : IsUniformInducing ⇑e
          cont : Continuous (⋯.extend ⇑f)
          eq : ∀ (b : E), Eq (⋯.extend (⇑f) (e b)) (f b)
          x y : E
          ⊢ Eq (⋯.extend (⇑f) (HAdd.hAdd (e x) (e y))) (HAdd.hAdd (⋯.extend (⇑f) (e x))  …
        -/
        simp only [eq, ← e.map_add]
        /-
          case refine_2
          𝕜 : Type u_1
          𝕜₂ : Type u_2
          E : Type u_3
          F : Type u_4
          Fₗ : Type u_5
          inst✝⁸ : NormedAddCommGroup E
          inst✝⁷ : NormedAddCommGroup F
          inst✝⁶ : NormedAddCommGroup Fₗ
          inst✝⁵ : NontriviallyNormedField 𝕜
          inst✝⁴ : NontriviallyNormedField 𝕜₂
          inst✝³ : NormedSpace 𝕜 E
          inst✝² : NormedSpace 𝕜₂ F
          inst✝¹ : NormedSpace 𝕜 Fₗ
          σ₁₂ : RingHom 𝕜 𝕜₂
          f g : ContinuousLinearMap σ₁₂ E F
          inst✝ : CompleteSpace F
          e : ContinuousLinearMap (RingHom.id 𝕜) E Fₗ
          h_dense : DenseRange ⇑e
          h_e : IsUniformInducing ⇑e
          cont : Continuous (⋯.extend ⇑f)
          eq : ∀ (b : E), Eq (⋯.extend (⇑f) (e b)) (f b)
          x y : E
          ⊢ Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        -/
        exact f.map_add _ _
        /-
          🎉 no goals
        -/
    map_smul' := fun k => by
      /-
        𝕜 : Type u_1
        𝕜₂ : Type u_2
        E : Type u_3
        F : Type u_4
        Fₗ : Type u_5
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : NormedAddCommGroup F
        inst✝⁶ : NormedAddCommGroup Fₗ
        inst✝⁵ : NontriviallyNormedField 𝕜
        inst✝⁴ : NontriviallyNormedField 𝕜₂
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedSpace 𝕜₂ F
        inst✝¹ : NormedSpace 𝕜 Fₗ
        σ₁₂ : RingHom 𝕜 𝕜₂
        f g : ContinuousLinearMap σ₁₂ E F
        inst✝ : CompleteSpace F
        e : ContinuousLinearMap (RingHom.id 𝕜) E Fₗ
        h_dense : DenseRange ⇑e
        h_e : IsUniformInducing ⇑e
        cont : Continuous (⋯.extend ⇑f)
        eq : ∀ (b : E), Eq (⋯.extend (⇑f) (e b)) (f b)
        k : 𝕜
        ⊢ ∀ (x : Fₗ), Eq ({ toFun := ⋯.extend ⇑f, map_add' := ⋯ }.toFun (HSMul.hSMul k …
      -/
      refine fun b => h_dense.induction_on b ?_ ?_
      · exact isClosed_eq (cont.comp (continuous_const_smul _))
          ((continuous_const_smul _).comp cont)
        /-
          case refine_2
          𝕜 : Type u_1
          𝕜₂ : Type u_2
          E : Type u_3
          F : Type u_4
          Fₗ : Type u_5
          inst✝⁸ : NormedAddCommGroup E
          inst✝⁷ : NormedAddCommGroup F
          inst✝⁶ : NormedAddCommGroup Fₗ
          inst✝⁵ : NontriviallyNormedField 𝕜
          inst✝⁴ : NontriviallyNormedField 𝕜₂
          inst✝³ : NormedSpace 𝕜 E
          inst✝² : NormedSpace 𝕜₂ F
          inst✝¹ : NormedSpace 𝕜 Fₗ
          σ₁₂ : RingHom 𝕜 𝕜₂
          f g : ContinuousLinearMap σ₁₂ E F
          inst✝ : CompleteSpace F
          e : ContinuousLinearMap (RingHom.id 𝕜) E Fₗ
          h_dense : DenseRange ⇑e
          h_e : IsUniformInducing ⇑e
          cont : Continuous (⋯.extend ⇑f)
          eq : ∀ (b : E), Eq (⋯.extend (⇑f) (e b)) (f b)
          k : 𝕜
          b : Fₗ
          ⊢ ∀ (a : E), Eq ({ toFun := ⋯.extend ⇑f, map_add' := ⋯ }.toFun (HSMul.hSMul k  …
        -/
      · intro x
        /-
          case refine_2
          𝕜 : Type u_1
          𝕜₂ : Type u_2
          E : Type u_3
          F : Type u_4
          Fₗ : Type u_5
          inst✝⁸ : NormedAddCommGroup E
          inst✝⁷ : NormedAddCommGroup F
          inst✝⁶ : NormedAddCommGroup Fₗ
          inst✝⁵ : NontriviallyNormedField 𝕜
          inst✝⁴ : NontriviallyNormedField 𝕜₂
          inst✝³ : NormedSpace 𝕜 E
          inst✝² : NormedSpace 𝕜₂ F
          inst✝¹ : NormedSpace 𝕜 Fₗ
          σ₁₂ : RingHom 𝕜 𝕜₂
          f g : ContinuousLinearMap σ₁₂ E F
          inst✝ : CompleteSpace F
          e : ContinuousLinearMap (RingHom.id 𝕜) E Fₗ
          h_dense : DenseRange ⇑e
          h_e : IsUniformInducing ⇑e
          cont : Continuous (⋯.extend ⇑f)
          eq : ∀ (b : E), Eq (⋯.extend (⇑f) (e b)) (f b)
          k : 𝕜
          b : Fₗ
          x : E
          ⊢ Eq ({ toFun := ⋯.extend ⇑f, map_add' := ⋯ }.toFun (HSMul.hSMul k (e x))) (HS …
        -/
        rw [← map_smul]
        /-
          case refine_2
          𝕜 : Type u_1
          𝕜₂ : Type u_2
          E : Type u_3
          F : Type u_4
          Fₗ : Type u_5
          inst✝⁸ : NormedAddCommGroup E
          inst✝⁷ : NormedAddCommGroup F
          inst✝⁶ : NormedAddCommGroup Fₗ
          inst✝⁵ : NontriviallyNormedField 𝕜
          inst✝⁴ : NontriviallyNormedField 𝕜₂
          inst✝³ : NormedSpace 𝕜 E
          inst✝² : NormedSpace 𝕜₂ F
          inst✝¹ : NormedSpace 𝕜 Fₗ
          σ₁₂ : RingHom 𝕜 𝕜₂
          f g : ContinuousLinearMap σ₁₂ E F
          inst✝ : CompleteSpace F
          e : ContinuousLinearMap (RingHom.id 𝕜) E Fₗ
          h_dense : DenseRange ⇑e
          h_e : IsUniformInducing ⇑e
          cont : Continuous (⋯.extend ⇑f)
          eq : ∀ (b : E), Eq (⋯.extend (⇑f) (e b)) (f b)
          k : 𝕜
          b : Fₗ
          x : E
          ⊢ Eq ({ toFun := ⋯.extend ⇑f, map_add' := ⋯ }.toFun (e (HSMul.hSMul k x))) (HS …
        -/
        simp only [eq]
        /-
          case refine_2
          𝕜 : Type u_1
          𝕜₂ : Type u_2
          E : Type u_3
          F : Type u_4
          Fₗ : Type u_5
          inst✝⁸ : NormedAddCommGroup E
          inst✝⁷ : NormedAddCommGroup F
          inst✝⁶ : NormedAddCommGroup Fₗ
          inst✝⁵ : NontriviallyNormedField 𝕜
          inst✝⁴ : NontriviallyNormedField 𝕜₂
          inst✝³ : NormedSpace 𝕜 E
          inst✝² : NormedSpace 𝕜₂ F
          inst✝¹ : NormedSpace 𝕜 Fₗ
          σ₁₂ : RingHom 𝕜 𝕜₂
          f g : ContinuousLinearMap σ₁₂ E F
          inst✝ : CompleteSpace F
          e : ContinuousLinearMap (RingHom.id 𝕜) E Fₗ
          h_dense : DenseRange ⇑e
          h_e : IsUniformInducing ⇑e
          cont : Continuous (⋯.extend ⇑f)
          eq : ∀ (b : E), Eq (⋯.extend (⇑f) (e b)) (f b)
          k : 𝕜
          b : Fₗ
          x : E
          ⊢ Eq (f (HSMul.hSMul k x)) (HSMul.hSMul (σ₁₂ k) (f x))
        -/
        exact ContinuousLinearMap.map_smulₛₗ _ _ _
        /-
          🎉 no goals
        -/
    cont }

-- Porting note: previously `(h_e.isDenseInducing h_dense)` was inferred.

@[simp]
theorem extend_eq (x : E) : extend f e h_dense h_e (e x) = f x :=
  IsDenseInducing.extend_eq (h_e.isDenseInducing h_dense) f.cont _


theorem extend_unique (g : Fₗ →SL[σ₁₂] F) (H : g.comp e = f) : extend f e h_dense h_e = g :=
  ContinuousLinearMap.coeFn_injective <|
    uniformly_extend_unique h_e h_dense (ContinuousLinearMap.ext_iff.1 H) g.continuous


@[simp]
theorem extend_zero : extend (0 : E →SL[σ₁₂] F) e h_dense h_e = 0 :=
  extend_unique _ _ _ _ _ (zero_comp _)


/-- If a dense embedding `e : E →L[𝕜] G` expands the norm by a constant factor `N⁻¹`, then the
norm of the extension of `f` along `e` is bounded by `N * ‖f‖`. -/
theorem opNorm_extend_le :
    ‖f.extend e h_dense (isUniformEmbedding_of_bound _ h_e).isUniformInducing‖ ≤ N * ‖f‖ := by
  -- Add `opNorm_le_of_dense`?
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_3
    F : Type u_4
    Fₗ : Type u_5
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedAddCommGroup Fₗ
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NontriviallyNormedField 𝕜₂
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedSpace 𝕜₂ F
    inst✝² : NormedSpace 𝕜 Fₗ
    σ₁₂ : RingHom 𝕜 𝕜₂
    f : ContinuousLinearMap σ₁₂ E F
    inst✝¹ : CompleteSpace F
    e : ContinuousLinearMap (RingHom.id 𝕜) E Fₗ
    h_dense : DenseRange ⇑e
    N : NNReal
    h_e : ∀ (x : E), LE.le (Norm.norm x) (HMul.hMul (↑N) (Norm.norm (e x)))
    inst✝ : RingHomIsometric σ₁₂
    ⊢ LE.le (Norm.norm (f.extend e h_dense ⋯)) (HMul.hMul (↑N) (Norm.norm f))
  -/
  refine opNorm_le_bound _ ?_ (isClosed_property h_dense (isClosed_le ?_ ?_) fun x ↦ ?_)
  · cases le_total 0 N with
    | inl hN => exact mul_nonneg hN (norm_nonneg _)
    | inr hN =>
      have : Unique E := ⟨⟨0⟩, fun x ↦ norm_le_zero_iff.mp <|
        (h_e x).trans (mul_nonpos_of_nonpos_of_nonneg hN (norm_nonneg _))⟩
      obtain rfl : f = 0 := Subsingleton.elim ..
      simp
    /-
      case refine_2
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_3
      F : Type u_4
      Fₗ : Type u_5
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedAddCommGroup Fₗ
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NontriviallyNormedField 𝕜₂
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedSpace 𝕜₂ F
      inst✝² : NormedSpace 𝕜 Fₗ
      σ₁₂ : RingHom 𝕜 𝕜₂
      f : ContinuousLinearMap σ₁₂ E F
      inst✝¹ : CompleteSpace F
      e : ContinuousLinearMap (RingHom.id 𝕜) E Fₗ
      h_dense : DenseRange ⇑e
      N : NNReal
      h_e : ∀ (x : E), LE.le (Norm.norm x) (HMul.hMul (↑N) (Norm.norm (e x)))
      inst✝ : RingHomIsometric σ₁₂
      ⊢ Continuous fun x => Norm.norm ((f.extend e h_dense ⋯) x)
    -/
  · exact (cont _).norm
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_3
      F : Type u_4
      Fₗ : Type u_5
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedAddCommGroup Fₗ
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NontriviallyNormedField 𝕜₂
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedSpace 𝕜₂ F
      inst✝² : NormedSpace 𝕜 Fₗ
      σ₁₂ : RingHom 𝕜 𝕜₂
      f : ContinuousLinearMap σ₁₂ E F
      inst✝¹ : CompleteSpace F
      e : ContinuousLinearMap (RingHom.id 𝕜) E Fₗ
      h_dense : DenseRange ⇑e
      N : NNReal
      h_e : ∀ (x : E), LE.le (Norm.norm x) (HMul.hMul (↑N) (Norm.norm (e x)))
      inst✝ : RingHomIsometric σ₁₂
      ⊢ Continuous fun x => HMul.hMul (HMul.hMul (↑N) (Norm.norm f)) (Norm.norm x)
    -/
  · exact continuous_const.mul continuous_norm
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_3
      F : Type u_4
      Fₗ : Type u_5
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedAddCommGroup F
      inst✝⁷ : NormedAddCommGroup Fₗ
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NontriviallyNormedField 𝕜₂
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedSpace 𝕜₂ F
      inst✝² : NormedSpace 𝕜 Fₗ
      σ₁₂ : RingHom 𝕜 𝕜₂
      f : ContinuousLinearMap σ₁₂ E F
      inst✝¹ : CompleteSpace F
      e : ContinuousLinearMap (RingHom.id 𝕜) E Fₗ
      h_dense : DenseRange ⇑e
      N : NNReal
      h_e : ∀ (x : E), LE.le (Norm.norm x) (HMul.hMul (↑N) (Norm.norm (e x)))
      inst✝ : RingHomIsometric σ₁₂
      x : E
      ⊢ LE.le (Norm.norm ((f.extend e h_dense ⋯) (e x))) (HMul.hMul (HMul.hMul (↑N)  …
    -/
  · rw [extend_eq]
    calc
      ‖f x‖ ≤ ‖f‖ * ‖x‖ := le_opNorm _ _
      _ ≤ ‖f‖ * (N * ‖e x‖) := mul_le_mul_of_nonneg_left (h_e x) (norm_nonneg _)
      _ ≤ N * ‖f‖ * ‖e x‖ := by rw [mul_comm ↑N ‖f‖, mul_assoc]


@[deprecated (since := "2024-02-02")] alias op_norm_extend_le := opNorm_extend_le



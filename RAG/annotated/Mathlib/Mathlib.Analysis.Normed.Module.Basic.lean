/-- A normed space over a normed field is a vector space endowed with a norm which satisfies the
equality `‖c • x‖ = ‖c‖ ‖x‖`. We require only `‖c • x‖ ≤ ‖c‖ ‖x‖` in the definition, then prove
`‖c • x‖ = ‖c‖ ‖x‖` in `norm_smul`.

Note that since this requires `SeminormedAddCommGroup` and not `NormedAddCommGroup`, this
typeclass can be used for "semi normed spaces" too, just as `Module` can be used for
"semi modules". -/
class NormedSpace (𝕜 : Type*) (E : Type*) [NormedField 𝕜] [SeminormedAddCommGroup E]
    extends Module 𝕜 E where
  norm_smul_le : ∀ (a : 𝕜) (b : E), ‖a • b‖ ≤ ‖a‖ * ‖b‖


instance (priority := 100) NormedSpace.boundedSMul [NormedSpace 𝕜 E] : BoundedSMul 𝕜 E :=
  BoundedSMul.of_norm_smul_le NormedSpace.norm_smul_le


instance NormedField.toNormedSpace : NormedSpace 𝕜 𝕜 where norm_smul_le a b := norm_mul_le a b

-- shortcut instance

instance NormedField.to_boundedSMul : BoundedSMul 𝕜 𝕜 :=
  NormedSpace.boundedSMul


variable (𝕜) in
theorem norm_zsmul (n : ℤ) (x : E) : ‖n • x‖ = ‖(n : 𝕜)‖ * ‖x‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    n : Int
    x : E
    ⊢ Eq (Norm.norm (HSMul.hSMul n x)) (HMul.hMul (Norm.norm ↑n) (Norm.norm x))
  -/
  rw [← norm_smul, ← Int.smul_one_eq_cast, smul_assoc, one_smul]
  /-
    🎉 no goals
  -/


theorem eventually_nhds_norm_smul_sub_lt (c : 𝕜) (x : E) {ε : ℝ} (h : 0 < ε) :
    ∀ᶠ y in 𝓝 x, ‖c • (y - x)‖ < ε :=
  have : Tendsto (fun y ↦ ‖c • (y - x)‖) (𝓝 x) (𝓝 0) :=
                            /-
                              𝕜 : Type u_1
                              E : Type u_3
                              inst✝² : NormedField 𝕜
                              inst✝¹ : SeminormedAddCommGroup E
                              inst✝ : NormedSpace 𝕜 E
                              c : 𝕜
                              x : E
                              ε : Real
                              h : LT.lt 0 ε
                              ⊢ Continuous fun y => Norm.norm (HSMul.hSMul c (HSub.hSub y x))
                            -/
                            /-
                              🎉 no goals
                            -/
    Continuous.tendsto' (by fun_prop) _ _ (by simp)
                                              /-
                                                🎉 no goals
                                              -/
  this.eventually (gt_mem_nhds h)


theorem Filter.Tendsto.zero_smul_isBoundedUnder_le {f : α → 𝕜} {g : α → E} {l : Filter α}
    (hf : Tendsto f l (𝓝 0)) (hg : IsBoundedUnder (· ≤ ·) l (Norm.norm ∘ g)) :
    Tendsto (fun x => f x • g x) l (𝓝 0) :=
  hf.op_zero_isBoundedUnder_le hg (· • ·) norm_smul_le


theorem Filter.IsBoundedUnder.smul_tendsto_zero {f : α → 𝕜} {g : α → E} {l : Filter α}
    (hf : IsBoundedUnder (· ≤ ·) l (norm ∘ f)) (hg : Tendsto g l (𝓝 0)) :
    Tendsto (fun x => f x • g x) l (𝓝 0) :=
  hg.op_zero_isBoundedUnder_le hf (flip (· • ·)) fun x y =>
    (norm_smul_le y x).trans_eq (mul_comm _ _)


instance NormedSpace.discreteTopology_zmultiples
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℚ E] (e : E) :
    DiscreteTopology <| AddSubgroup.zmultiples e := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    E✝ : Type u_3
    F : Type u_4
    α : Type u_5
    inst✝⁶ : NormedField 𝕜
    inst✝⁵ : SeminormedAddCommGroup E✝
    inst✝⁴ : SeminormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 E✝
    inst✝² : NormedSpace 𝕜 F
    E : Type u_6
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Rat E
    e : E
    ⊢ DiscreteTopology (Subtype fun x => Membership.mem (AddSubgroup.zmultiples e) …
  -/
  rcases eq_or_ne e 0 with (rfl | he)
    /-
      case inl
      𝕜 : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      inst✝⁶ : NormedField 𝕜
      inst✝⁵ : SeminormedAddCommGroup E✝
      inst✝⁴ : SeminormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 E✝
      inst✝² : NormedSpace 𝕜 F
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Rat E
      ⊢ DiscreteTopology (Subtype fun x => Membership.mem (AddSubgroup.zmultiples 0) …
    -/
  · rw [AddSubgroup.zmultiples_zero_eq_bot]
    /-
      case inl
      𝕜 : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      inst✝⁶ : NormedField 𝕜
      inst✝⁵ : SeminormedAddCommGroup E✝
      inst✝⁴ : SeminormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 E✝
      inst✝² : NormedSpace 𝕜 F
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Rat E
      ⊢ DiscreteTopology (Subtype fun x => Membership.mem Bot.bot x)
    -/
    exact Subsingleton.discreteTopology (α := ↑(⊥ : Subspace ℚ E))
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      inst✝⁶ : NormedField 𝕜
      inst✝⁵ : SeminormedAddCommGroup E✝
      inst✝⁴ : SeminormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 E✝
      inst✝² : NormedSpace 𝕜 F
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Rat E
      e : E
      he : Ne e 0
      ⊢ DiscreteTopology (Subtype fun x => Membership.mem (AddSubgroup.zmultiples e) …
    -/
  · rw [discreteTopology_iff_isOpen_singleton_zero, isOpen_induced_iff]
    /-
      case inr
      𝕜 : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      inst✝⁶ : NormedField 𝕜
      inst✝⁵ : SeminormedAddCommGroup E✝
      inst✝⁴ : SeminormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 E✝
      inst✝² : NormedSpace 𝕜 F
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Rat E
      e : E
      he : Ne e 0
      ⊢ Exists fun t => And (IsOpen t) (Eq (Set.preimage Subtype.val t) (Singleton.s …
    -/
    refine ⟨Metric.ball 0 ‖e‖, Metric.isOpen_ball, ?_⟩
    /-
      case inr
      𝕜 : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      inst✝⁶ : NormedField 𝕜
      inst✝⁵ : SeminormedAddCommGroup E✝
      inst✝⁴ : SeminormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 E✝
      inst✝² : NormedSpace 𝕜 F
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Rat E
      e : E
      he : Ne e 0
      ⊢ Eq (Set.preimage Subtype.val (Metric.ball 0 (Norm.norm e))) (Singleton.singl …
    -/
    ext ⟨x, hx⟩
    /-
      case inr.h.mk
      𝕜 : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      inst✝⁶ : NormedField 𝕜
      inst✝⁵ : SeminormedAddCommGroup E✝
      inst✝⁴ : SeminormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 E✝
      inst✝² : NormedSpace 𝕜 F
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Rat E
      e : E
      he : Ne e 0
      x : E
      hx : Membership.mem (AddSubgroup.zmultiples e) x
      ⊢ Iff (Membership.mem (Set.preimage Subtype.val (Metric.ball 0 (Norm.norm e))) …
    -/
    obtain ⟨k, rfl⟩ := AddSubgroup.mem_zmultiples_iff.mp hx
    rw [mem_preimage, mem_ball_zero_iff, AddSubgroup.coe_mk, mem_singleton_iff, Subtype.ext_iff,
      AddSubgroup.coe_mk, AddSubgroup.coe_zero, norm_zsmul ℚ k e, Int.norm_cast_rat,
      Int.norm_eq_abs, mul_lt_iff_lt_one_left (norm_pos_iff.mpr he), ← @Int.cast_one ℝ _,
      ← Int.cast_abs, Int.cast_lt, Int.abs_lt_one_iff, smul_eq_zero, or_iff_left he]


instance ULift.normedSpace : NormedSpace 𝕜 (ULift E) :=
  { __ := ULift.seminormedAddCommGroup (E := E),
    __ := ULift.module'
    norm_smul_le := fun s x => (norm_smul_le s x.down : _) }


/-- The product of two normed spaces is a normed space, with the sup norm. -/
instance Prod.normedSpace : NormedSpace 𝕜 (E × F) :=
  { Prod.seminormedAddCommGroup (E := E) (F := F), Prod.instModule with
    norm_smul_le := fun s x => by
      simp only [norm_smul, Prod.norm_def, Prod.smul_snd, Prod.smul_fst,
        mul_max_of_nonneg, norm_nonneg, le_rfl] }


/-- The product of finitely many normed spaces is a normed space, with the sup norm. -/
instance Pi.normedSpace {ι : Type*} {E : ι → Type*} [Fintype ι] [∀ i, SeminormedAddCommGroup (E i)]
    [∀ i, NormedSpace 𝕜 (E i)] : NormedSpace 𝕜 (∀ i, E i) where
  norm_smul_le a f := by
    simp_rw [← coe_nnnorm, ← NNReal.coe_mul, NNReal.coe_le_coe, Pi.nnnorm_def,
      NNReal.mul_finset_sup]
    /-
      𝕜 : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      inst✝⁷ : NormedField 𝕜
      inst✝⁶ : SeminormedAddCommGroup E✝
      inst✝⁵ : SeminormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 E✝
      inst✝³ : NormedSpace 𝕜 F
      ι : Type u_6
      E : ι → Type u_7
      inst✝² : Fintype ι
      inst✝¹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
      a : 𝕜
      f : (i : ι) → E i
      ⊢ LE.le (Finset.univ.sup fun b => NNNorm.nnnorm (HSMul.hSMul a f b)) (Finset.u …
    -/
    exact Finset.sup_mono_fun fun _ _ => norm_smul_le a _
    /-
      🎉 no goals
    -/


instance SeparationQuotient.instNormedSpace : NormedSpace 𝕜 (SeparationQuotient E) where
  norm_smul_le := norm_smul_le


instance MulOpposite.instNormedSpace : NormedSpace 𝕜 Eᵐᵒᵖ where
  norm_smul_le _ x := norm_smul_le _ x.unop


/-- A subspace of a normed space is also a normed space, with the restriction of the norm. -/
instance Submodule.normedSpace {𝕜 R : Type*} [SMul 𝕜 R] [NormedField 𝕜] [Ring R] {E : Type*}
    [SeminormedAddCommGroup E] [NormedSpace 𝕜 E] [Module R E] [IsScalarTower 𝕜 R E]
    (s : Submodule R E) : NormedSpace 𝕜 s where
  norm_smul_le c x := norm_smul_le c (x : E)


instance (priority := 75) SubmoduleClass.toNormedSpace : NormedSpace 𝕜 s where
  norm_smul_le c x := norm_smul_le c (x : E)


/-- A linear map from a `Module` to a `NormedSpace` induces a `NormedSpace` structure on the
domain, using the `SeminormedAddCommGroup.induced` norm.

See note [reducible non-instances] -/
abbrev NormedSpace.induced {F : Type*} (𝕜 E G : Type*) [NormedField 𝕜] [AddCommGroup E] [Module 𝕜 E]
    [SeminormedAddCommGroup G] [NormedSpace 𝕜 G] [FunLike F E G] [LinearMapClass F 𝕜 E G] (f : F) :
    @NormedSpace 𝕜 E _ (SeminormedAddCommGroup.induced E G f) :=
  let _ := SeminormedAddCommGroup.induced E G f
                /-
                  𝕜✝ : Type u_1
                  𝕜' : Type u_2
                  E✝ : Type u_3
                  F✝ : Type u_4
                  α : Type u_5
                  F : Type u_6
                  𝕜 : Type u_7
                  E : Type u_8
                  G : Type u_9
                  inst✝⁶ : NormedField 𝕜
                  inst✝⁵ : AddCommGroup E
                  inst✝⁴ : Module 𝕜 E
                  inst✝³ : SeminormedAddCommGroup G
                  inst✝² : NormedSpace 𝕜 G
                  inst✝¹ : FunLike F E G
                  inst✝ : LinearMapClass F 𝕜 E G
                  f : F
                  x✝ : SeminormedAddCommGroup E := SeminormedAddCommGroup.induced E G f
                  a : 𝕜
                  b : E
                  ⊢ LE.le (Norm.norm (HSMul.hSMul a b)) (HMul.hMul (Norm.norm a) (Norm.norm b))
                -/
  ⟨fun a b ↦ by simpa only [← map_smul f a b] using norm_smul_le a (f b)⟩
                /-
                  🎉 no goals
                -/


/-- If `E` is a nontrivial normed space over a nontrivially normed field `𝕜`, then `E` is unbounded:
for any `c : ℝ`, there exists a vector `x : E` with norm strictly greater than `c`. -/
theorem NormedSpace.exists_lt_norm (c : ℝ) : ∃ x : E, c < ‖x‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : Nontrivial E
    c : Real
    ⊢ Exists fun x => LT.lt c (Norm.norm x)
  -/
  rcases exists_ne (0 : E) with ⟨x, hx⟩
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : Nontrivial E
    c : Real
    x : E
    hx : Ne x 0
    ⊢ Exists fun x => LT.lt c (Norm.norm x)
  -/
  rcases NormedField.exists_lt_norm 𝕜 (c / ‖x‖) with ⟨r, hr⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : Nontrivial E
    c : Real
    x : E
    hx : Ne x 0
    r : 𝕜
    hr : LT.lt (HDiv.hDiv c (Norm.norm x)) (Norm.norm r)
    ⊢ Exists fun x => LT.lt c (Norm.norm x)
  -/
  use r • x
  /-
    case h
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : Nontrivial E
    c : Real
    x : E
    hx : Ne x 0
    r : 𝕜
    hr : LT.lt (HDiv.hDiv c (Norm.norm x)) (Norm.norm r)
    ⊢ LT.lt c (Norm.norm (HSMul.hSMul r x))
  -/
  rwa [norm_smul, ← div_lt_iff₀]
  /-
    case h
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : Nontrivial E
    c : Real
    x : E
    hx : Ne x 0
    r : 𝕜
    hr : LT.lt (HDiv.hDiv c (Norm.norm x)) (Norm.norm r)
    ⊢ LT.lt 0 (Norm.norm x)
  -/
  rwa [norm_pos_iff]
  /-
    🎉 no goals
  -/


protected theorem NormedSpace.unbounded_univ : ¬Bornology.IsBounded (univ : Set E) := fun h =>
  let ⟨R, hR⟩ := isBounded_iff_forall_norm_le.1 h
  let ⟨x, hx⟩ := NormedSpace.exists_lt_norm 𝕜 E R
  hx.not_le (hR x trivial)


protected lemma NormedSpace.cobounded_neBot : NeBot (cobounded E) := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : Nontrivial E
    ⊢ (Bornology.cobounded E).NeBot
  -/
  rw [neBot_iff, Ne, cobounded_eq_bot_iff, ← isBounded_univ]
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : Nontrivial E
    ⊢ Not (Bornology.IsBounded Set.univ)
  -/
  exact NormedSpace.unbounded_univ 𝕜 E
  /-
    🎉 no goals
  -/


instance (priority := 100) NontriviallyNormedField.cobounded_neBot : NeBot (cobounded 𝕜) :=
  NormedSpace.cobounded_neBot 𝕜 𝕜


instance (priority := 80) RealNormedSpace.cobounded_neBot [NormedSpace ℝ E] :
    NeBot (cobounded E) := NormedSpace.cobounded_neBot ℝ E


instance (priority := 80) NontriviallyNormedField.infinite : Infinite 𝕜 :=
  ⟨fun _ ↦ NormedSpace.unbounded_univ 𝕜 𝕜 (Set.toFinite _).isBounded⟩


/-- A normed vector space over an infinite normed field is a noncompact space.
This cannot be an instance because in order to apply it,
Lean would have to search for `NormedSpace 𝕜 E` with unknown `𝕜`.
We register this as an instance in two cases: `𝕜 = E` and `𝕜 = ℝ`. -/
protected theorem NormedSpace.noncompactSpace : NoncompactSpace E := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    inst✝⁴ : NormedField 𝕜
    inst✝³ : Infinite 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : Nontrivial E
    inst✝ : NormedSpace 𝕜 E
    ⊢ NoncompactSpace E
  -/
  by_cases H : ∃ c : 𝕜, c ≠ 0 ∧ ‖c‖ ≠ 1
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_3
      inst✝⁴ : NormedField 𝕜
      inst✝³ : Infinite 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : Nontrivial E
      inst✝ : NormedSpace 𝕜 E
      H : Exists fun c => And (Ne c 0) (Ne (Norm.norm c) 1)
      ⊢ NoncompactSpace E
    -/
  · letI := NontriviallyNormedField.ofNormNeOne H
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_3
      inst✝⁴ : NormedField 𝕜
      inst✝³ : Infinite 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : Nontrivial E
      inst✝ : NormedSpace 𝕜 E
      H : Exists fun c => And (Ne c 0) (Ne (Norm.norm c) 1)
      this : NontriviallyNormedField 𝕜 := NontriviallyNormedField.ofNormNeOne H
      ⊢ NoncompactSpace E
    -/
    exact ⟨fun h ↦ NormedSpace.unbounded_univ 𝕜 E h.isBounded⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      E : Type u_3
      inst✝⁴ : NormedField 𝕜
      inst✝³ : Infinite 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : Nontrivial E
      inst✝ : NormedSpace 𝕜 E
      H : Not (Exists fun c => And (Ne c 0) (Ne (Norm.norm c) 1))
      ⊢ NoncompactSpace E
    -/
  · push_neg at H
    /-
      case neg
      𝕜 : Type u_1
      E : Type u_3
      inst✝⁴ : NormedField 𝕜
      inst✝³ : Infinite 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : Nontrivial E
      inst✝ : NormedSpace 𝕜 E
      H : ∀ (c : 𝕜), Ne c 0 → Eq (Norm.norm c) 1
      ⊢ NoncompactSpace E
    -/
    rcases exists_ne (0 : E) with ⟨x, hx⟩
    /-
      case neg.intro
      𝕜 : Type u_1
      E : Type u_3
      inst✝⁴ : NormedField 𝕜
      inst✝³ : Infinite 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : Nontrivial E
      inst✝ : NormedSpace 𝕜 E
      H : ∀ (c : 𝕜), Ne c 0 → Eq (Norm.norm c) 1
      x : E
      hx : Ne x 0
      ⊢ NoncompactSpace E
    -/
    suffices IsClosedEmbedding (Infinite.natEmbedding 𝕜 · • x) from this.noncompactSpace
    /-
      case neg.intro
      𝕜 : Type u_1
      E : Type u_3
      inst✝⁴ : NormedField 𝕜
      inst✝³ : Infinite 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : Nontrivial E
      inst✝ : NormedSpace 𝕜 E
      H : ∀ (c : 𝕜), Ne c 0 → Eq (Norm.norm c) 1
      x : E
      hx : Ne x 0
      ⊢ Topology.IsClosedEmbedding fun x_1 => HSMul.hSMul ((Infinite.natEmbedding 𝕜) …
    -/
    refine isClosedEmbedding_of_pairwise_le_dist (norm_pos_iff.2 hx) fun k n hne ↦ ?_
    /-
      case neg.intro
      𝕜 : Type u_1
      E : Type u_3
      inst✝⁴ : NormedField 𝕜
      inst✝³ : Infinite 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : Nontrivial E
      inst✝ : NormedSpace 𝕜 E
      H : ∀ (c : 𝕜), Ne c 0 → Eq (Norm.norm c) 1
      x : E
      hx : Ne x 0
      k n : Nat
      hne : Ne k n
      ⊢ (fun x_1 y => LE.le (Norm.norm x) (Dist.dist (HSMul.hSMul ((Infinite.natEmbe …
    -/
    simp only [dist_eq_norm, ← sub_smul, norm_smul]
    /-
      case neg.intro
      𝕜 : Type u_1
      E : Type u_3
      inst✝⁴ : NormedField 𝕜
      inst✝³ : Infinite 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : Nontrivial E
      inst✝ : NormedSpace 𝕜 E
      H : ∀ (c : 𝕜), Ne c 0 → Eq (Norm.norm c) 1
      x : E
      hx : Ne x 0
      k n : Nat
      hne : Ne k n
      ⊢ LE.le (Norm.norm x) (HMul.hMul (Norm.norm (HSub.hSub ((Infinite.natEmbedding …
    -/
    rw [H, one_mul]
    /-
      case neg.intro._
      𝕜 : Type u_1
      E : Type u_3
      inst✝⁴ : NormedField 𝕜
      inst✝³ : Infinite 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : Nontrivial E
      inst✝ : NormedSpace 𝕜 E
      H : ∀ (c : 𝕜), Ne c 0 → Eq (Norm.norm c) 1
      x : E
      hx : Ne x 0
      k n : Nat
      hne : Ne k n
      ⊢ Ne (HSub.hSub ((Infinite.natEmbedding 𝕜) k) ((Infinite.natEmbedding 𝕜) n)) 0
    -/
    rwa [sub_ne_zero, (Embedding.injective _).ne_iff]
    /-
      🎉 no goals
    -/


instance (priority := 100) NormedField.noncompactSpace : NoncompactSpace 𝕜 :=
  NormedSpace.noncompactSpace 𝕜 𝕜


instance (priority := 100) RealNormedSpace.noncompactSpace [NormedSpace ℝ E] : NoncompactSpace E :=
  NormedSpace.noncompactSpace ℝ E


/-- A normed algebra `𝕜'` over `𝕜` is normed module that is also an algebra.

See the implementation notes for `Algebra` for a discussion about non-unital algebras. Following
the strategy there, a non-unital *normed* algebra can be written as:
```lean
variable [NormedField 𝕜] [NonUnitalSeminormedRing 𝕜']
variable [NormedSpace 𝕜 𝕜'] [SMulCommClass 𝕜 𝕜' 𝕜'] [IsScalarTower 𝕜 𝕜' 𝕜']
```
-/
class NormedAlgebra (𝕜 : Type*) (𝕜' : Type*) [NormedField 𝕜] [SeminormedRing 𝕜'] extends
  Algebra 𝕜 𝕜' where
  norm_smul_le : ∀ (r : 𝕜) (x : 𝕜'), ‖r • x‖ ≤ ‖r‖ * ‖x‖


instance (priority := 100) NormedAlgebra.toNormedSpace : NormedSpace 𝕜 𝕜' :=
  -- Porting note: previous Lean could figure out what we were extending
  { NormedAlgebra.toAlgebra.toModule with
  norm_smul_le := NormedAlgebra.norm_smul_le }


theorem norm_algebraMap (x : 𝕜) : ‖algebraMap 𝕜 𝕜' x‖ = ‖x‖ * ‖(1 : 𝕜')‖ := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedRing 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    x : 𝕜
    ⊢ Eq (Norm.norm ((algebraMap 𝕜 𝕜') x)) (HMul.hMul (Norm.norm x) (Norm.norm 1))
  -/
  rw [Algebra.algebraMap_eq_smul_one]
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedRing 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    x : 𝕜
    ⊢ Eq (Norm.norm (HSMul.hSMul x 1)) (HMul.hMul (Norm.norm x) (Norm.norm 1))
  -/
  exact norm_smul _ _
  /-
    🎉 no goals
  -/


theorem nnnorm_algebraMap (x : 𝕜) : ‖algebraMap 𝕜 𝕜' x‖₊ = ‖x‖₊ * ‖(1 : 𝕜')‖₊ :=
  Subtype.ext <| norm_algebraMap 𝕜' x


theorem dist_algebraMap (x y : 𝕜) :
    (dist (algebraMap 𝕜 𝕜' x) (algebraMap 𝕜 𝕜' y)) = dist x y * ‖(1 : 𝕜')‖ := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedRing 𝕜'
    inst✝ : NormedAlgebra 𝕜 𝕜'
    x y : 𝕜
    ⊢ Eq (Dist.dist ((algebraMap 𝕜 𝕜') x) ((algebraMap 𝕜 𝕜') y)) (HMul.hMul (Dist. …
  -/
  simp only [dist_eq_norm, ← map_sub, norm_algebraMap]
  /-
    🎉 no goals
  -/


/-- This is a simpler version of `norm_algebraMap` when `‖1‖ = 1` in `𝕜'`.-/
@[simp]
theorem norm_algebraMap' [NormOneClass 𝕜'] (x : 𝕜) : ‖algebraMap 𝕜 𝕜' x‖ = ‖x‖ := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : SeminormedRing 𝕜'
    inst✝¹ : NormedAlgebra 𝕜 𝕜'
    inst✝ : NormOneClass 𝕜'
    x : 𝕜
    ⊢ Eq (Norm.norm ((algebraMap 𝕜 𝕜') x)) (Norm.norm x)
  -/
  rw [norm_algebraMap, norm_one, mul_one]
  /-
    🎉 no goals
  -/


/-- This is a simpler version of `nnnorm_algebraMap` when `‖1‖ = 1` in `𝕜'`.-/
@[simp]
theorem nnnorm_algebraMap' [NormOneClass 𝕜'] (x : 𝕜) : ‖algebraMap 𝕜 𝕜' x‖₊ = ‖x‖₊ :=
  Subtype.ext <| norm_algebraMap' _ _


/-- This is a simpler version of `dist_algebraMap` when `‖1‖ = 1` in `𝕜'`.-/
@[simp]
theorem dist_algebraMap' [NormOneClass 𝕜'] (x y : 𝕜) :
    (dist (algebraMap 𝕜 𝕜' x) (algebraMap 𝕜 𝕜' y)) = dist x y := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : SeminormedRing 𝕜'
    inst✝¹ : NormedAlgebra 𝕜 𝕜'
    inst✝ : NormOneClass 𝕜'
    x y : 𝕜
    ⊢ Eq (Dist.dist ((algebraMap 𝕜 𝕜') x) ((algebraMap 𝕜 𝕜') y)) (Dist.dist x y)
  -/
  simp only [dist_eq_norm, ← map_sub, norm_algebraMap']
  /-
    🎉 no goals
  -/


@[simp]
theorem norm_algebraMap_nnreal (x : ℝ≥0) : ‖algebraMap ℝ≥0 𝕜' x‖ = x :=
  (norm_algebraMap' 𝕜' (x : ℝ)).symm ▸ Real.norm_of_nonneg x.prop


@[simp]
theorem nnnorm_algebraMap_nnreal (x : ℝ≥0) : ‖algebraMap ℝ≥0 𝕜' x‖₊ = x :=
  Subtype.ext <| norm_algebraMap_nnreal 𝕜' x


/-- In a normed algebra, the inclusion of the base field in the extended field is an isometry. -/
theorem algebraMap_isometry [NormOneClass 𝕜'] : Isometry (algebraMap 𝕜 𝕜') := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : SeminormedRing 𝕜'
    inst✝¹ : NormedAlgebra 𝕜 𝕜'
    inst✝ : NormOneClass 𝕜'
    ⊢ Isometry ⇑(algebraMap 𝕜 𝕜')
  -/
  refine Isometry.of_dist_eq fun x y => ?_
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : SeminormedRing 𝕜'
    inst✝¹ : NormedAlgebra 𝕜 𝕜'
    inst✝ : NormOneClass 𝕜'
    x y : 𝕜
    ⊢ Eq (Dist.dist ((algebraMap 𝕜 𝕜') x) ((algebraMap 𝕜 𝕜') y)) (Dist.dist x y)
  -/
  rw [dist_eq_norm, dist_eq_norm, ← RingHom.map_sub, norm_algebraMap']
  /-
    🎉 no goals
  -/


instance NormedAlgebra.id : NormedAlgebra 𝕜 𝕜 :=
  { NormedField.toNormedSpace, Algebra.id 𝕜 with }

-- Porting note: cannot synth scalar tower ℚ ℝ k

/-- Any normed characteristic-zero division ring that is a normed algebra over the reals is also a
normed algebra over the rationals.

Phrased another way, if `𝕜` is a normed algebra over the reals, then `AlgebraRat` respects that
norm. -/
instance normedAlgebraRat {𝕜} [NormedDivisionRing 𝕜] [CharZero 𝕜] [NormedAlgebra ℝ 𝕜] :
    NormedAlgebra ℚ 𝕜 where
  norm_smul_le q x := by
    /-
      𝕜✝ : Type u_1
      𝕜' : Type u_2
      E : Type u_3
      F : Type u_4
      α : Type u_5
      inst✝⁵ : NormedField 𝕜✝
      inst✝⁴ : SeminormedRing 𝕜'
      inst✝³ : NormedAlgebra 𝕜✝ 𝕜'
      𝕜 : Type ?u.93596
      inst✝² : NormedDivisionRing 𝕜
      inst✝¹ : CharZero 𝕜
      inst✝ : NormedAlgebra Real 𝕜
      q : Rat
      x : 𝕜
      ⊢ LE.le (Norm.norm (HSMul.hSMul q x)) (HMul.hMul (Norm.norm q) (Norm.norm x))
    -/
    rw [← smul_one_smul ℝ q x, Rat.smul_one_eq_cast, norm_smul, Rat.norm_cast_real]
    /-
      🎉 no goals
    -/


instance PUnit.normedAlgebra : NormedAlgebra 𝕜 PUnit where
                         /-
                           𝕜 : Type u_1
                           𝕜' : Type u_2
                           E : Type u_3
                           F : Type u_4
                           α : Type u_5
                           inst✝² : NormedField 𝕜
                           inst✝¹ : SeminormedRing 𝕜'
                           inst✝ : NormedAlgebra 𝕜 𝕜'
                           q : 𝕜
                           x✝ : PUnit.{?u.96883 + 1}
                           ⊢ LE.le (Norm.norm (HSMul.hSMul q x✝)) (HMul.hMul (Norm.norm q) (Norm.norm x✝))
                         -/
  norm_smul_le q _ := by simp only [norm_eq_zero, mul_zero, le_refl]
                         /-
                           🎉 no goals
                         -/


instance : NormedAlgebra 𝕜 (ULift 𝕜') :=
  { ULift.normedSpace, ULift.algebra with }


/-- The product of two normed algebras is a normed algebra, with the sup norm. -/
instance Prod.normedAlgebra {E F : Type*} [SeminormedRing E] [SeminormedRing F] [NormedAlgebra 𝕜 E]
    [NormedAlgebra 𝕜 F] : NormedAlgebra 𝕜 (E × F) :=
  { Prod.normedSpace, Prod.algebra 𝕜 E F with }

-- Porting note: Lean 3 could synth the algebra instances for Pi Pr

/-- The product of finitely many normed algebras is a normed algebra, with the sup norm. -/
instance Pi.normedAlgebra {ι : Type*} {E : ι → Type*} [Fintype ι] [∀ i, SeminormedRing (E i)]
    [∀ i, NormedAlgebra 𝕜 (E i)] : NormedAlgebra 𝕜 (∀ i, E i) :=
  { Pi.normedSpace, Pi.algebra _ E with }


instance SeparationQuotient.instNormedAlgebra : NormedAlgebra 𝕜 (SeparationQuotient E) where
  __ : NormedSpace 𝕜 (SeparationQuotient E) := inferInstance
  __ : Algebra 𝕜 (SeparationQuotient E) := inferInstance


instance MulOpposite.instNormedAlgebra {E : Type*} [SeminormedRing E] [NormedAlgebra 𝕜 E] :
    NormedAlgebra 𝕜 Eᵐᵒᵖ where
  __ := instAlgebra
  __ := instNormedSpace


/-- A non-unital algebra homomorphism from an `Algebra` to a `NormedAlgebra` induces a
`NormedAlgebra` structure on the domain, using the `SeminormedRing.induced` norm.

See note [reducible non-instances] -/
abbrev NormedAlgebra.induced {F : Type*} (𝕜 R S : Type*) [NormedField 𝕜] [Ring R] [Algebra 𝕜 R]
    [SeminormedRing S] [NormedAlgebra 𝕜 S] [FunLike F R S] [NonUnitalAlgHomClass F 𝕜 R S]
    (f : F) :
    @NormedAlgebra 𝕜 R _ (SeminormedRing.induced R S f) :=
  letI := SeminormedRing.induced R S f
  ⟨fun a b ↦ show ‖f (a • b)‖ ≤ ‖a‖ * ‖f b‖ from (map_smul f a b).symm ▸ norm_smul_le a (f b)⟩

-- Porting note: failed to synth NonunitalAlgHomClass

instance Subalgebra.toNormedAlgebra {𝕜 A : Type*} [SeminormedRing A] [NormedField 𝕜]
    [NormedAlgebra 𝕜 A] (S : Subalgebra 𝕜 A) : NormedAlgebra 𝕜 S :=
  NormedAlgebra.induced 𝕜 S A S.val


instance (priority := 75) SubalgebraClass.toNormedAlgebra : NormedAlgebra 𝕜 s where
  norm_smul_le c x := norm_smul_le c (x : E)


instance [I : SeminormedAddCommGroup E] :
    SeminormedAddCommGroup (RestrictScalars 𝕜 𝕜' E) :=
  I


instance [I : NormedAddCommGroup E] :
    NormedAddCommGroup (RestrictScalars 𝕜 𝕜' E) :=
  I


instance [I : NonUnitalSeminormedRing E] :
    NonUnitalSeminormedRing (RestrictScalars 𝕜 𝕜' E) :=
  I


instance [I : NonUnitalNormedRing E] :
    NonUnitalNormedRing (RestrictScalars 𝕜 𝕜' E) :=
  I


instance [I : SeminormedRing E] :
    SeminormedRing (RestrictScalars 𝕜 𝕜' E) :=
  I


instance [I : NormedRing E] :
    NormedRing (RestrictScalars 𝕜 𝕜' E) :=
  I


instance [I : NonUnitalSeminormedCommRing E] :
    NonUnitalSeminormedCommRing (RestrictScalars 𝕜 𝕜' E) :=
  I


instance [I : NonUnitalNormedCommRing E] :
    NonUnitalNormedCommRing (RestrictScalars 𝕜 𝕜' E) :=
  I


instance [I : SeminormedCommRing E] :
    SeminormedCommRing (RestrictScalars 𝕜 𝕜' E) :=
  I


instance [I : NormedCommRing E] :
    NormedCommRing (RestrictScalars 𝕜 𝕜' E) :=
  I


/-- If `E` is a normed space over `𝕜'` and `𝕜` is a normed algebra over `𝕜'`, then
`RestrictScalars.module` is additionally a `NormedSpace`. -/
instance RestrictScalars.normedSpace : NormedSpace 𝕜 (RestrictScalars 𝕜 𝕜' E) :=
  { RestrictScalars.module 𝕜 𝕜' E with
    norm_smul_le := fun c x =>
                                                                /-
                                                                  𝕜 : Type u_1
                                                                  𝕜' : Type u_2
                                                                  E : Type u_3
                                                                  F : Type u_4
                                                                  α : Type u_5
                                                                  inst✝⁴ : NormedField 𝕜
                                                                  inst✝³ : NormedField 𝕜'
                                                                  inst✝² : NormedAlgebra 𝕜 𝕜'
                                                                  inst✝¹ : SeminormedAddCommGroup E
                                                                  inst✝ : NormedSpace 𝕜' E
                                                                  c : 𝕜
                                                                  x : RestrictScalars 𝕜 𝕜' E
                                                                  ⊢ Eq (HMul.hMul (Norm.norm ((algebraMap 𝕜 𝕜') c)) (Norm.norm x)) (HMul.hMul (N …
                                                                -/
      (norm_smul_le (algebraMap 𝕜 𝕜' c) (_ : E)).trans_eq <| by rw [norm_algebraMap'] }
                                                                /-
                                                                  🎉 no goals
                                                                -/

-- If you think you need this, consider instead reproducing `RestrictScalars.lsmul`
-- appropriately modified here.

/-- The action of the original normed_field on `RestrictScalars 𝕜 𝕜' E`.
This is not an instance as it would be contrary to the purpose of `RestrictScalars`.
-/
def Module.RestrictScalars.normedSpaceOrig {𝕜 : Type*} {𝕜' : Type*} {E : Type*} [NormedField 𝕜']
    [SeminormedAddCommGroup E] [I : NormedSpace 𝕜' E] : NormedSpace 𝕜' (RestrictScalars 𝕜 𝕜' E) :=
  I


/-- Warning: This declaration should be used judiciously.
Please consider using `IsScalarTower` and/or `RestrictScalars 𝕜 𝕜' E` instead.

This definition allows the `RestrictScalars.normedSpace` instance to be put directly on `E`
rather on `RestrictScalars 𝕜 𝕜' E`. This would be a very bad instance; both because `𝕜'` cannot be
inferred, and because it is likely to create instance diamonds.

See Note [reducible non-instances].
-/
abbrev NormedSpace.restrictScalars : NormedSpace 𝕜 E :=
  RestrictScalars.normedSpace _ 𝕜' E


/-- If `E` is a normed algebra over `𝕜'` and `𝕜` is a normed algebra over `𝕜'`, then
`RestrictScalars.module` is additionally a `NormedAlgebra`. -/
instance RestrictScalars.normedAlgebra : NormedAlgebra 𝕜 (RestrictScalars 𝕜 𝕜' E) :=
  { RestrictScalars.algebra 𝕜 𝕜' E with
    norm_smul_le := norm_smul_le }

-- If you think you need this, consider instead reproducing `RestrictScalars.lsmul`
-- appropriately modified here.

/-- The action of the original normed_field on `RestrictScalars 𝕜 𝕜' E`.
This is not an instance as it would be contrary to the purpose of `RestrictScalars`.
-/
def Module.RestrictScalars.normedAlgebraOrig {𝕜 : Type*} {𝕜' : Type*} {E : Type*} [NormedField 𝕜']
    [SeminormedRing E] [I : NormedAlgebra 𝕜' E] : NormedAlgebra 𝕜' (RestrictScalars 𝕜 𝕜' E) :=
  I


/-- Warning: This declaration should be used judiciously.
Please consider using `IsScalarTower` and/or `RestrictScalars 𝕜 𝕜' E` instead.

This definition allows the `RestrictScalars.normedAlgebra` instance to be put directly on `E`
rather on `RestrictScalars 𝕜 𝕜' E`. This would be a very bad instance; both because `𝕜'` cannot be
inferred, and because it is likely to create instance diamonds.

See Note [reducible non-instances].
-/
abbrev NormedAlgebra.restrictScalars : NormedAlgebra 𝕜 E :=
  RestrictScalars.normedAlgebra _ 𝕜' _


/-- A structure encapsulating minimal axioms needed to defined a seminormed vector space, as found
in textbooks. This is meant to be used to easily define `SeminormedAddCommGroup E` instances from
scratch on a type with no preexisting distance or topology. -/
structure SeminormedAddCommGroup.Core (𝕜 : Type*) (E : Type*) [NormedField 𝕜] [AddCommGroup E]
    [Norm E] [Module 𝕜 E] : Prop where
  norm_nonneg (x : E) : 0 ≤ ‖x‖
  norm_smul (c : 𝕜) (x : E) : ‖c • x‖ = ‖c‖ * ‖x‖
  norm_triangle (x y : E) : ‖x + y‖ ≤ ‖x‖ + ‖y‖


/-- Produces a `PseudoMetricSpace E` instance from a `SeminormedAddCommGroup.Core`. Note that
if this is used to define an instance on a type, it also provides a new uniformity and
topology on the type. See note [reducible non-instances]. -/
abbrev PseudoMetricSpace.ofSeminormedAddCommGroupCore {𝕜 E : Type*} [NormedField 𝕜] [AddCommGroup E]
    [Norm E] [Module 𝕜 E] (core : SeminormedAddCommGroup.Core 𝕜 E) :
    PseudoMetricSpace E where
  dist x y := ‖x - y‖
  dist_self x := by
    /-
      𝕜✝ : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      𝕜 : Type u_6
      E : Type u_7
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Norm E
      inst✝ : Module 𝕜 E
      core : SeminormedAddCommGroup.Core 𝕜 E
      x : E
      ⊢ Eq (Dist.dist x x) 0
    -/
    show ‖x - x‖ = 0
    /-
      𝕜✝ : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      𝕜 : Type u_6
      E : Type u_7
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Norm E
      inst✝ : Module 𝕜 E
      core : SeminormedAddCommGroup.Core 𝕜 E
      x : E
      ⊢ Eq (Norm.norm (HSub.hSub x x)) 0
    -/
    simp only [sub_self]
    /-
      𝕜✝ : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      𝕜 : Type u_6
      E : Type u_7
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Norm E
      inst✝ : Module 𝕜 E
      core : SeminormedAddCommGroup.Core 𝕜 E
      x : E
      ⊢ Eq (Norm.norm 0) 0
    -/
    have : (0 : E) = (0 : 𝕜) • (0 : E) := by simp
    /-
      𝕜✝ : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      𝕜 : Type u_6
      E : Type u_7
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Norm E
      inst✝ : Module 𝕜 E
      core : SeminormedAddCommGroup.Core 𝕜 E
      x : E
      this : Eq 0 (HSMul.hSMul 0 0)
      ⊢ Eq (Norm.norm 0) 0
    -/
    rw [this, core.norm_smul]
    /-
      𝕜✝ : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      𝕜 : Type u_6
      E : Type u_7
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Norm E
      inst✝ : Module 𝕜 E
      core : SeminormedAddCommGroup.Core 𝕜 E
      x : E
      this : Eq 0 (HSMul.hSMul 0 0)
      ⊢ Eq (HMul.hMul (Norm.norm 0) (Norm.norm 0)) 0
    -/
    simp
    /-
      🎉 no goals
    -/
  dist_comm x y := by
    /-
      𝕜✝ : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      𝕜 : Type u_6
      E : Type u_7
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Norm E
      inst✝ : Module 𝕜 E
      core : SeminormedAddCommGroup.Core 𝕜 E
      x y : E
      ⊢ Eq (Dist.dist x y) (Dist.dist y x)
    -/
    show ‖x - y‖ = ‖y - x‖
    /-
      𝕜✝ : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      𝕜 : Type u_6
      E : Type u_7
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Norm E
      inst✝ : Module 𝕜 E
      core : SeminormedAddCommGroup.Core 𝕜 E
      x y : E
      ⊢ Eq (Norm.norm (HSub.hSub x y)) (Norm.norm (HSub.hSub y x))
    -/
    have : y - x = (-1 : 𝕜) • (x - y) := by simp
    /-
      𝕜✝ : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      𝕜 : Type u_6
      E : Type u_7
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Norm E
      inst✝ : Module 𝕜 E
      core : SeminormedAddCommGroup.Core 𝕜 E
      x y : E
      this : Eq (HSub.hSub y x) (HSMul.hSMul (-1) (HSub.hSub x y))
      ⊢ Eq (Norm.norm (HSub.hSub x y)) (Norm.norm (HSub.hSub y x))
    -/
    rw [this, core.norm_smul]
    /-
      𝕜✝ : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      𝕜 : Type u_6
      E : Type u_7
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Norm E
      inst✝ : Module 𝕜 E
      core : SeminormedAddCommGroup.Core 𝕜 E
      x y : E
      this : Eq (HSub.hSub y x) (HSMul.hSMul (-1) (HSub.hSub x y))
      ⊢ Eq (Norm.norm (HSub.hSub x y)) (HMul.hMul (Norm.norm (-1)) (Norm.norm (HSub. …
    -/
    simp
    /-
      🎉 no goals
    -/
  dist_triangle x y z := by
    /-
      𝕜✝ : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      𝕜 : Type u_6
      E : Type u_7
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Norm E
      inst✝ : Module 𝕜 E
      core : SeminormedAddCommGroup.Core 𝕜 E
      x y z : E
      ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
    -/
    show ‖x - z‖ ≤ ‖x - y‖ + ‖y - z‖
    /-
      𝕜✝ : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      𝕜 : Type u_6
      E : Type u_7
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Norm E
      inst✝ : Module 𝕜 E
      core : SeminormedAddCommGroup.Core 𝕜 E
      x y z : E
      ⊢ LE.le (Norm.norm (HSub.hSub x z)) (HAdd.hAdd (Norm.norm (HSub.hSub x y)) (No …
    -/
    have : x - z = (x - y) + (y - z) := by abel
    /-
      𝕜✝ : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      𝕜 : Type u_6
      E : Type u_7
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Norm E
      inst✝ : Module 𝕜 E
      core : SeminormedAddCommGroup.Core 𝕜 E
      x y z : E
      this : Eq (HSub.hSub x z) (HAdd.hAdd (HSub.hSub x y) (HSub.hSub y z))
      ⊢ LE.le (Norm.norm (HSub.hSub x z)) (HAdd.hAdd (Norm.norm (HSub.hSub x y)) (No …
    -/
    rw [this]
    /-
      𝕜✝ : Type u_1
      𝕜' : Type u_2
      E✝ : Type u_3
      F : Type u_4
      α : Type u_5
      𝕜 : Type u_6
      E : Type u_7
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Norm E
      inst✝ : Module 𝕜 E
      core : SeminormedAddCommGroup.Core 𝕜 E
      x y z : E
      this : Eq (HSub.hSub x z) (HAdd.hAdd (HSub.hSub x y) (HSub.hSub y z))
      ⊢ LE.le (Norm.norm (HAdd.hAdd (HSub.hSub x y) (HSub.hSub y z))) (HAdd.hAdd (No …
    -/
    exact core.norm_triangle _ _
    /-
      🎉 no goals
    -/
                       /-
                         𝕜✝ : Type u_1
                         𝕜' : Type u_2
                         E✝ : Type u_3
                         F : Type u_4
                         α : Type u_5
                         𝕜 : Type u_6
                         E : Type u_7
                         inst✝³ : NormedField 𝕜
                         inst✝² : AddCommGroup E
                         inst✝¹ : Norm E
                         inst✝ : Module 𝕜 E
                         core : SeminormedAddCommGroup.Core 𝕜 E
                         x y : E
                         ⊢ Eq ((fun x y => ↑⟨Norm.norm (HSub.hSub x y), ⋯⟩) x y) (ENNReal.ofReal (Dist. …
                       -/
  edist_dist x y := by exact (ENNReal.ofReal_eq_coe_nnreal _).symm
                       /-
                         🎉 no goals
                       -/


/-- Produces a `PseudoEMetricSpace E` instance from a `SeminormedAddCommGroup.Core`. Note that
if this is used to define an instance on a type, it also provides a new uniformity and
topology on the type. See note [reducible non-instances]. -/
abbrev PseudoEMetricSpace.ofSeminormedAddCommGroupCore {𝕜 E : Type*} [NormedField 𝕜]
    [AddCommGroup E] [Norm E] [Module 𝕜 E]
    (core : SeminormedAddCommGroup.Core 𝕜 E) : PseudoEMetricSpace E :=
  (PseudoMetricSpace.ofSeminormedAddCommGroupCore core).toPseudoEMetricSpace


/-- Produces a `PseudoEMetricSpace E` instance from a `SeminormedAddCommGroup.Core` on a type that
already has an existing uniform space structure. This requires a proof that the uniformity induced
by the norm is equal to the preexisting uniformity. See note [reducible non-instances]. -/
abbrev PseudoMetricSpace.ofSeminormedAddCommGroupCoreReplaceUniformity {𝕜 E : Type*} [NormedField 𝕜]
    [AddCommGroup E] [Norm E] [Module 𝕜 E] [U : UniformSpace E]
    (core : SeminormedAddCommGroup.Core 𝕜 E)
    (H : 𝓤[U] = 𝓤[PseudoEMetricSpace.toUniformSpace
        (self := PseudoEMetricSpace.ofSeminormedAddCommGroupCore core)]) :
    PseudoMetricSpace E :=
  .replaceUniformity (.ofSeminormedAddCommGroupCore core) H


open Bornology in
/-- Produces a `PseudoEMetricSpace E` instance from a `SeminormedAddCommGroup.Core` on a type that
already has a preexisting uniform space structure and a preexisting bornology. This requires proofs
that the uniformity induced by the norm is equal to the preexisting uniformity, and likewise for
the bornology. See note [reducible non-instances]. -/
abbrev PseudoMetricSpace.ofSeminormedAddCommGroupCoreReplaceAll {𝕜 E : Type*} [NormedField 𝕜]
    [AddCommGroup E] [Norm E] [Module 𝕜 E] [U : UniformSpace E] [B : Bornology E]
    (core : SeminormedAddCommGroup.Core 𝕜 E)
    (HU : 𝓤[U] = 𝓤[PseudoEMetricSpace.toUniformSpace
      (self := PseudoEMetricSpace.ofSeminormedAddCommGroupCore core)])
    (HB : ∀ s : Set E, @IsBounded _ B s
      ↔ @IsBounded _ (PseudoMetricSpace.ofSeminormedAddCommGroupCore core).toBornology s) :
    PseudoMetricSpace E :=
  .replaceBornology (.replaceUniformity (.ofSeminormedAddCommGroupCore core) HU) HB


/-- Produces a `SeminormedAddCommGroup E` instance from a `SeminormedAddCommGroup.Core`. Note that
if this is used to define an instance on a type, it also provides a new distance measure from the
norm.  it must therefore not be used on a type with a preexisting distance measure or topology.
See note [reducible non-instances]. -/
abbrev SeminormedAddCommGroup.ofCore {𝕜 : Type*} {E : Type*} [NormedField 𝕜] [AddCommGroup E]
    [Norm E] [Module 𝕜 E] (core : SeminormedAddCommGroup.Core 𝕜 E) : SeminormedAddCommGroup E :=
  { PseudoMetricSpace.ofSeminormedAddCommGroupCore core with }


/-- Produces a `SeminormedAddCommGroup E` instance from a `SeminormedAddCommGroup.Core` on a type
that already has an existing uniform space structure. This requires a proof that the uniformity
induced by the norm is equal to the preexisting uniformity. See note [reducible non-instances]. -/
abbrev SeminormedAddCommGroup.ofCoreReplaceUniformity {𝕜 : Type*} {E : Type*} [NormedField 𝕜]
    [AddCommGroup E] [Norm E] [Module 𝕜 E] [U : UniformSpace E]
    (core : SeminormedAddCommGroup.Core 𝕜 E)
    (H : 𝓤[U] = 𝓤[PseudoEMetricSpace.toUniformSpace
      (self := PseudoEMetricSpace.ofSeminormedAddCommGroupCore core)]) :
    SeminormedAddCommGroup E :=
  { PseudoMetricSpace.ofSeminormedAddCommGroupCoreReplaceUniformity core H with }


open Bornology in
/-- Produces a `SeminormedAddCommGroup E` instance from a `SeminormedAddCommGroup.Core` on a type
that already has a preexisting uniform space structure and a preexisting bornology. This requires
proofs that the uniformity induced by the norm is equal to the preexisting uniformity, and likewise
for the bornology. See note [reducible non-instances]. -/
abbrev SeminormedAddCommGroup.ofCoreReplaceAll {𝕜 : Type*} {E : Type*} [NormedField 𝕜]
    [AddCommGroup E] [Norm E] [Module 𝕜 E] [U : UniformSpace E] [B : Bornology E]
    (core : SeminormedAddCommGroup.Core 𝕜 E)
    (HU : 𝓤[U] = 𝓤[PseudoEMetricSpace.toUniformSpace
      (self := PseudoEMetricSpace.ofSeminormedAddCommGroupCore core)])
    (HB : ∀ s : Set E, @IsBounded _ B s
      ↔ @IsBounded _ (PseudoMetricSpace.ofSeminormedAddCommGroupCore core).toBornology s) :
    SeminormedAddCommGroup E :=
  { PseudoMetricSpace.ofSeminormedAddCommGroupCoreReplaceAll core HU HB with }


/-- A structure encapsulating minimal axioms needed to defined a normed vector space, as found
in textbooks. This is meant to be used to easily define `NormedAddCommGroup E` and `NormedSpace E`
instances from scratch on a type with no preexisting distance or topology. -/
structure NormedSpace.Core (𝕜 : Type*) (E : Type*) [NormedField 𝕜] [AddCommGroup E] [Module 𝕜 E]
    [Norm E] extends SeminormedAddCommGroup.Core 𝕜 E : Prop where
  norm_eq_zero_iff (x : E) : ‖x‖ = 0 ↔ x = 0


/-- Produces a `NormedAddCommGroup E` instance from a `NormedSpace.Core`. Note that if this is
used to define an instance on a type, it also provides a new distance measure from the norm.
it must therefore not be used on a type with a preexisting distance measure.
See note [reducible non-instances]. -/
abbrev NormedAddCommGroup.ofCore (core : NormedSpace.Core 𝕜 E) : NormedAddCommGroup E :=
  { SeminormedAddCommGroup.ofCore core.toCore with
    eq_of_dist_eq_zero := by
      /-
        𝕜✝ : Type u_1
        𝕜' : Type u_2
        E✝ : Type u_3
        F : Type u_4
        α : Type u_5
        𝕜 : Type u_6
        E : Type u_7
        inst✝³ : NormedField 𝕜
        inst✝² : AddCommGroup E
        inst✝¹ : Module 𝕜 E
        inst✝ : Norm E
        core : NormedSpace.Core 𝕜 E
        ⊢ ∀ {x y : E}, Eq (Dist.dist x y) 0 → Eq x y
      -/
      intro x y h
      /-
        𝕜✝ : Type u_1
        𝕜' : Type u_2
        E✝ : Type u_3
        F : Type u_4
        α : Type u_5
        𝕜 : Type u_6
        E : Type u_7
        inst✝³ : NormedField 𝕜
        inst✝² : AddCommGroup E
        inst✝¹ : Module 𝕜 E
        inst✝ : Norm E
        core : NormedSpace.Core 𝕜 E
        x y : E
        h : Eq (Dist.dist x y) 0
        ⊢ Eq x y
      -/
      rw [← sub_eq_zero, ← core.norm_eq_zero_iff]
      /-
        𝕜✝ : Type u_1
        𝕜' : Type u_2
        E✝ : Type u_3
        F : Type u_4
        α : Type u_5
        𝕜 : Type u_6
        E : Type u_7
        inst✝³ : NormedField 𝕜
        inst✝² : AddCommGroup E
        inst✝¹ : Module 𝕜 E
        inst✝ : Norm E
        core : NormedSpace.Core 𝕜 E
        x y : E
        h : Eq (Dist.dist x y) 0
        ⊢ Eq (Norm.norm (HSub.hSub x y)) 0
      -/
      exact h }
      /-
        🎉 no goals
      -/


/-- Produces a `NormedAddCommGroup E` instance from a `NormedAddCommGroup.Core` on a type
that already has an existing uniform space structure. This requires a proof that the uniformity
induced by the norm is equal to the preexisting uniformity. See note [reducible non-instances]. -/
abbrev NormedAddCommGroup.ofCoreReplaceUniformity [U : UniformSpace E] (core : NormedSpace.Core 𝕜 E)
    (H : 𝓤[U] = 𝓤[PseudoEMetricSpace.toUniformSpace
      (self := PseudoEMetricSpace.ofSeminormedAddCommGroupCore core.toCore)]) :
    NormedAddCommGroup E :=
  { SeminormedAddCommGroup.ofCoreReplaceUniformity core.toCore H with
    eq_of_dist_eq_zero := by
      /-
        𝕜✝ : Type u_1
        𝕜' : Type u_2
        E✝ : Type u_3
        F : Type u_4
        α : Type u_5
        𝕜 : Type u_6
        E : Type u_7
        inst✝³ : NormedField 𝕜
        inst✝² : AddCommGroup E
        inst✝¹ : Module 𝕜 E
        inst✝ : Norm E
        U : UniformSpace E
        core : NormedSpace.Core 𝕜 E
        H : Eq (uniformity E) (uniformity E)
        ⊢ ∀ {x y : E}, Eq (Dist.dist x y) 0 → Eq x y
      -/
      intro x y h
      /-
        𝕜✝ : Type u_1
        𝕜' : Type u_2
        E✝ : Type u_3
        F : Type u_4
        α : Type u_5
        𝕜 : Type u_6
        E : Type u_7
        inst✝³ : NormedField 𝕜
        inst✝² : AddCommGroup E
        inst✝¹ : Module 𝕜 E
        inst✝ : Norm E
        U : UniformSpace E
        core : NormedSpace.Core 𝕜 E
        H : Eq (uniformity E) (uniformity E)
        x y : E
        h : Eq (Dist.dist x y) 0
        ⊢ Eq x y
      -/
      rw [← sub_eq_zero, ← core.norm_eq_zero_iff]
      /-
        𝕜✝ : Type u_1
        𝕜' : Type u_2
        E✝ : Type u_3
        F : Type u_4
        α : Type u_5
        𝕜 : Type u_6
        E : Type u_7
        inst✝³ : NormedField 𝕜
        inst✝² : AddCommGroup E
        inst✝¹ : Module 𝕜 E
        inst✝ : Norm E
        U : UniformSpace E
        core : NormedSpace.Core 𝕜 E
        H : Eq (uniformity E) (uniformity E)
        x y : E
        h : Eq (Dist.dist x y) 0
        ⊢ Eq (Norm.norm (HSub.hSub x y)) 0
      -/
      exact h }
      /-
        🎉 no goals
      -/


open Bornology in
/-- Produces a `NormedAddCommGroup E` instance from a `NormedAddCommGroup.Core` on a type
that already has a preexisting uniform space structure and a preexisting bornology. This requires
proofs that the uniformity induced by the norm is equal to the preexisting uniformity, and likewise
for the bornology. See note [reducible non-instances]. -/
abbrev NormedAddCommGroup.ofCoreReplaceAll [U : UniformSpace E] [B : Bornology E]
    (core : NormedSpace.Core 𝕜 E)
    (HU : 𝓤[U] = 𝓤[PseudoEMetricSpace.toUniformSpace
      (self := PseudoEMetricSpace.ofSeminormedAddCommGroupCore core.toCore)])
    (HB : ∀ s : Set E, @IsBounded _ B s
      ↔ @IsBounded _ (PseudoMetricSpace.ofSeminormedAddCommGroupCore core.toCore).toBornology s) :
    NormedAddCommGroup E :=
  { SeminormedAddCommGroup.ofCoreReplaceAll core.toCore HU HB with
    eq_of_dist_eq_zero := by
      /-
        𝕜✝ : Type u_1
        𝕜' : Type u_2
        E✝ : Type u_3
        F : Type u_4
        α : Type u_5
        𝕜 : Type u_6
        E : Type u_7
        inst✝³ : NormedField 𝕜
        inst✝² : AddCommGroup E
        inst✝¹ : Module 𝕜 E
        inst✝ : Norm E
        U : UniformSpace E
        B : Bornology E
        core : NormedSpace.Core 𝕜 E
        HU : Eq (uniformity E) (uniformity E)
        HB : ∀ (s : Set E), Iff (Bornology.IsBounded s) (Bornology.IsBounded s)
        ⊢ ∀ {x y : E}, Eq (Dist.dist x y) 0 → Eq x y
      -/
      intro x y h
      /-
        𝕜✝ : Type u_1
        𝕜' : Type u_2
        E✝ : Type u_3
        F : Type u_4
        α : Type u_5
        𝕜 : Type u_6
        E : Type u_7
        inst✝³ : NormedField 𝕜
        inst✝² : AddCommGroup E
        inst✝¹ : Module 𝕜 E
        inst✝ : Norm E
        U : UniformSpace E
        B : Bornology E
        core : NormedSpace.Core 𝕜 E
        HU : Eq (uniformity E) (uniformity E)
        HB : ∀ (s : Set E), Iff (Bornology.IsBounded s) (Bornology.IsBounded s)
        x y : E
        h : Eq (Dist.dist x y) 0
        ⊢ Eq x y
      -/
      rw [← sub_eq_zero, ← core.norm_eq_zero_iff]
      /-
        𝕜✝ : Type u_1
        𝕜' : Type u_2
        E✝ : Type u_3
        F : Type u_4
        α : Type u_5
        𝕜 : Type u_6
        E : Type u_7
        inst✝³ : NormedField 𝕜
        inst✝² : AddCommGroup E
        inst✝¹ : Module 𝕜 E
        inst✝ : Norm E
        U : UniformSpace E
        B : Bornology E
        core : NormedSpace.Core 𝕜 E
        HU : Eq (uniformity E) (uniformity E)
        HB : ∀ (s : Set E), Iff (Bornology.IsBounded s) (Bornology.IsBounded s)
        x y : E
        h : Eq (Dist.dist x y) 0
        ⊢ Eq (Norm.norm (HSub.hSub x y)) 0
      -/
      exact h }
      /-
        🎉 no goals
      -/


/-- Produces a `NormedSpace 𝕜 E` instance from a `NormedSpace.Core`. This is meant to be used
on types where the `NormedAddCommGroup E` instance has also been defined using `core`.
See note [reducible non-instances]. -/
abbrev NormedSpace.ofCore {𝕜 : Type*} {E : Type*} [NormedField 𝕜] [SeminormedAddCommGroup E]
    [Module 𝕜 E] (core : NormedSpace.Core 𝕜 E) : NormedSpace 𝕜 E where
                         /-
                           𝕜✝¹ : Type u_1
                           𝕜' : Type u_2
                           E✝¹ : Type u_3
                           F : Type u_4
                           α : Type u_5
                           𝕜✝ : Type u_6
                           E✝ : Type u_7
                           inst✝⁶ : NormedField 𝕜✝
                           inst✝⁵ : AddCommGroup E✝
                           inst✝⁴ : Module 𝕜✝ E✝
                           inst✝³ : Norm E✝
                           𝕜 : Type u_8
                           E : Type u_9
                           inst✝² : NormedField 𝕜
                           inst✝¹ : SeminormedAddCommGroup E
                           inst✝ : Module 𝕜 E
                           core : NormedSpace.Core 𝕜 E
                           r : 𝕜
                           x : E
                           ⊢ LE.le (Norm.norm (HSMul.hSMul r x)) (HMul.hMul (Norm.norm r) (Norm.norm x))
                         -/
  norm_smul_le r x := by rw [core.norm_smul r x]
                         /-
                           🎉 no goals
                         -/



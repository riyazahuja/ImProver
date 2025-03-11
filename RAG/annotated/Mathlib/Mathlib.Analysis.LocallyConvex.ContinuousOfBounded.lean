/-- Construct a continuous linear map from a linear map `f : E →ₗ[𝕜] F` and the existence of a
neighborhood of zero that gets mapped into a bounded set in `F`. -/
def LinearMap.clmOfExistsBoundedImage (f : E →ₗ[𝕜] F)
    (h : ∃ V ∈ 𝓝 (0 : E), Bornology.IsVonNBounded 𝕜 (f '' V)) : E →L[𝕜] F :=
  ⟨f, by
    -- It suffices to show that `f` is continuous at `0`.
    /-
      𝕜 : Type u_1
      𝕜' : Type u_2
      E : Type u_3
      F : Type u_4
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : UniformSpace E
      inst✝⁷ : UniformAddGroup E
      inst✝⁶ : AddCommGroup F
      inst✝⁵ : UniformSpace F
      inst✝⁴ : UniformAddGroup F
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 F
      inst✝ : ContinuousSMul 𝕜 E
      f : LinearMap (RingHom.id 𝕜) E F
      h : Exists fun V => And (Membership.mem (nhds 0) V) (Bornology.IsVonNBounded 𝕜 …
      ⊢ Continuous f.toFun
    -/
    refine continuous_of_continuousAt_zero f ?_
    /-
      𝕜 : Type u_1
      𝕜' : Type u_2
      E : Type u_3
      F : Type u_4
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : UniformSpace E
      inst✝⁷ : UniformAddGroup E
      inst✝⁶ : AddCommGroup F
      inst✝⁵ : UniformSpace F
      inst✝⁴ : UniformAddGroup F
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 F
      inst✝ : ContinuousSMul 𝕜 E
      f : LinearMap (RingHom.id 𝕜) E F
      h : Exists fun V => And (Membership.mem (nhds 0) V) (Bornology.IsVonNBounded 𝕜 …
      ⊢ ContinuousAt (⇑f) 0
    -/
    rw [continuousAt_def, f.map_zero]
    /-
      𝕜 : Type u_1
      𝕜' : Type u_2
      E : Type u_3
      F : Type u_4
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : UniformSpace E
      inst✝⁷ : UniformAddGroup E
      inst✝⁶ : AddCommGroup F
      inst✝⁵ : UniformSpace F
      inst✝⁴ : UniformAddGroup F
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 F
      inst✝ : ContinuousSMul 𝕜 E
      f : LinearMap (RingHom.id 𝕜) E F
      h : Exists fun V => And (Membership.mem (nhds 0) V) (Bornology.IsVonNBounded 𝕜 …
      ⊢ ∀ (A : Set F), Membership.mem (nhds 0) A → Membership.mem (nhds 0) (Set.prei …
    -/
    intro U hU
    -- Continuity means that `U ∈ 𝓝 0` implies that `f ⁻¹' U ∈ 𝓝 0`.
    /-
      𝕜 : Type u_1
      𝕜' : Type u_2
      E : Type u_3
      F : Type u_4
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : UniformSpace E
      inst✝⁷ : UniformAddGroup E
      inst✝⁶ : AddCommGroup F
      inst✝⁵ : UniformSpace F
      inst✝⁴ : UniformAddGroup F
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 F
      inst✝ : ContinuousSMul 𝕜 E
      f : LinearMap (RingHom.id 𝕜) E F
      h : Exists fun V => And (Membership.mem (nhds 0) V) (Bornology.IsVonNBounded 𝕜 …
      U : Set F
      hU : Membership.mem (nhds 0) U
      ⊢ Membership.mem (nhds 0) (Set.preimage (⇑f) U)
    -/
    rcases h with ⟨V, hV, h⟩
    /-
      case intro.intro
      𝕜 : Type u_1
      𝕜' : Type u_2
      E : Type u_3
      F : Type u_4
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : UniformSpace E
      inst✝⁷ : UniformAddGroup E
      inst✝⁶ : AddCommGroup F
      inst✝⁵ : UniformSpace F
      inst✝⁴ : UniformAddGroup F
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 F
      inst✝ : ContinuousSMul 𝕜 E
      f : LinearMap (RingHom.id 𝕜) E F
      U : Set F
      hU : Membership.mem (nhds 0) U
      V : Set E
      hV : Membership.mem (nhds 0) V
      h : Bornology.IsVonNBounded 𝕜 (Set.image (⇑f) V)
      ⊢ Membership.mem (nhds 0) (Set.preimage (⇑f) U)
    -/
    rcases (h hU).exists_pos with ⟨r, hr, h⟩
    /-
      case intro.intro.intro.intro
      𝕜 : Type u_1
      𝕜' : Type u_2
      E : Type u_3
      F : Type u_4
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : UniformSpace E
      inst✝⁷ : UniformAddGroup E
      inst✝⁶ : AddCommGroup F
      inst✝⁵ : UniformSpace F
      inst✝⁴ : UniformAddGroup F
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 F
      inst✝ : ContinuousSMul 𝕜 E
      f : LinearMap (RingHom.id 𝕜) E F
      U : Set F
      hU : Membership.mem (nhds 0) U
      V : Set E
      hV : Membership.mem (nhds 0) V
      h✝ : Bornology.IsVonNBounded 𝕜 (Set.image (⇑f) V)
      r : Real
      hr : GT.gt r 0
      h : ∀ (c : 𝕜), LE.le r (Norm.norm c) → HasSubset.Subset (Set.image (⇑f) V) (HS …
      ⊢ Membership.mem (nhds 0) (Set.preimage (⇑f) U)
    -/
    rcases NormedField.exists_lt_norm 𝕜 r with ⟨x, hx⟩
    /-
      case intro.intro.intro.intro.intro
      𝕜 : Type u_1
      𝕜' : Type u_2
      E : Type u_3
      F : Type u_4
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : UniformSpace E
      inst✝⁷ : UniformAddGroup E
      inst✝⁶ : AddCommGroup F
      inst✝⁵ : UniformSpace F
      inst✝⁴ : UniformAddGroup F
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 F
      inst✝ : ContinuousSMul 𝕜 E
      f : LinearMap (RingHom.id 𝕜) E F
      U : Set F
      hU : Membership.mem (nhds 0) U
      V : Set E
      hV : Membership.mem (nhds 0) V
      h✝ : Bornology.IsVonNBounded 𝕜 (Set.image (⇑f) V)
      r : Real
      hr : GT.gt r 0
      h : ∀ (c : 𝕜), LE.le r (Norm.norm c) → HasSubset.Subset (Set.image (⇑f) V) (HS …
      x : 𝕜
      hx : LT.lt r (Norm.norm x)
      ⊢ Membership.mem (nhds 0) (Set.preimage (⇑f) U)
    -/
    specialize h x hx.le
    -- After unfolding all the definitions, we know that `f '' V ⊆ x • U`. We use this to show the
    -- inclusion `x⁻¹ • V ⊆ f⁻¹' U`.
    /-
      case intro.intro.intro.intro.intro
      𝕜 : Type u_1
      𝕜' : Type u_2
      E : Type u_3
      F : Type u_4
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : UniformSpace E
      inst✝⁷ : UniformAddGroup E
      inst✝⁶ : AddCommGroup F
      inst✝⁵ : UniformSpace F
      inst✝⁴ : UniformAddGroup F
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 F
      inst✝ : ContinuousSMul 𝕜 E
      f : LinearMap (RingHom.id 𝕜) E F
      U : Set F
      hU : Membership.mem (nhds 0) U
      V : Set E
      hV : Membership.mem (nhds 0) V
      h✝ : Bornology.IsVonNBounded 𝕜 (Set.image (⇑f) V)
      r : Real
      hr : GT.gt r 0
      x : 𝕜
      hx : LT.lt r (Norm.norm x)
      h : HasSubset.Subset (Set.image (⇑f) V) (HSMul.hSMul x U)
      ⊢ Membership.mem (nhds 0) (Set.preimage (⇑f) U)
    -/
    have x_ne := norm_pos_iff.mp (hr.trans hx)
    have : x⁻¹ • V ⊆ f ⁻¹' U :=
      calc
        x⁻¹ • V ⊆ x⁻¹ • f ⁻¹' (f '' V) := Set.smul_set_mono (Set.subset_preimage_image (⇑f) V)
        _ ⊆ x⁻¹ • f ⁻¹' (x • U) := Set.smul_set_mono (Set.preimage_mono h)
        _ = f ⁻¹' (x⁻¹ • x • U) := by
          ext
          simp only [Set.mem_inv_smul_set_iff₀ x_ne, Set.mem_preimage, LinearMap.map_smul]
        _ ⊆ f ⁻¹' U := by rw [inv_smul_smul₀ x_ne _]
    -- Using this inclusion, it suffices to show that `x⁻¹ • V` is in `𝓝 0`, which is trivial.
    /-
      case intro.intro.intro.intro.intro
      𝕜 : Type u_1
      𝕜' : Type u_2
      E : Type u_3
      F : Type u_4
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : UniformSpace E
      inst✝⁷ : UniformAddGroup E
      inst✝⁶ : AddCommGroup F
      inst✝⁵ : UniformSpace F
      inst✝⁴ : UniformAddGroup F
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 F
      inst✝ : ContinuousSMul 𝕜 E
      f : LinearMap (RingHom.id 𝕜) E F
      U : Set F
      hU : Membership.mem (nhds 0) U
      V : Set E
      hV : Membership.mem (nhds 0) V
      h✝ : Bornology.IsVonNBounded 𝕜 (Set.image (⇑f) V)
      r : Real
      hr : GT.gt r 0
      x : 𝕜
      hx : LT.lt r (Norm.norm x)
      h : HasSubset.Subset (Set.image (⇑f) V) (HSMul.hSMul x U)
      x_ne : Ne x 0
      this : HasSubset.Subset (HSMul.hSMul (Inv.inv x) V) (Set.preimage (⇑f) U)
      ⊢ Membership.mem (nhds 0) (Set.preimage (⇑f) U)
    -/
    refine mem_of_superset ?_ this
    /-
      case intro.intro.intro.intro.intro
      𝕜 : Type u_1
      𝕜' : Type u_2
      E : Type u_3
      F : Type u_4
      inst✝⁹ : AddCommGroup E
      inst✝⁸ : UniformSpace E
      inst✝⁷ : UniformAddGroup E
      inst✝⁶ : AddCommGroup F
      inst✝⁵ : UniformSpace F
      inst✝⁴ : UniformAddGroup F
      inst✝³ : NontriviallyNormedField 𝕜
      inst✝² : Module 𝕜 E
      inst✝¹ : Module 𝕜 F
      inst✝ : ContinuousSMul 𝕜 E
      f : LinearMap (RingHom.id 𝕜) E F
      U : Set F
      hU : Membership.mem (nhds 0) U
      V : Set E
      hV : Membership.mem (nhds 0) V
      h✝ : Bornology.IsVonNBounded 𝕜 (Set.image (⇑f) V)
      r : Real
      hr : GT.gt r 0
      x : 𝕜
      hx : LT.lt r (Norm.norm x)
      h : HasSubset.Subset (Set.image (⇑f) V) (HSMul.hSMul x U)
      x_ne : Ne x 0
      this : HasSubset.Subset (HSMul.hSMul (Inv.inv x) V) (Set.preimage (⇑f) U)
      ⊢ Membership.mem (nhds 0) (HSMul.hSMul (Inv.inv x) V)
    -/
    rwa [set_smul_mem_nhds_zero_iff (inv_ne_zero x_ne)]⟩
    /-
      🎉 no goals
    -/


theorem LinearMap.clmOfExistsBoundedImage_coe {f : E →ₗ[𝕜] F}
    {h : ∃ V ∈ 𝓝 (0 : E), Bornology.IsVonNBounded 𝕜 (f '' V)} :
    (f.clmOfExistsBoundedImage h : E →ₗ[𝕜] F) = f :=
  rfl


@[simp]
theorem LinearMap.clmOfExistsBoundedImage_apply {f : E →ₗ[𝕜] F}
    {h : ∃ V ∈ 𝓝 (0 : E), Bornology.IsVonNBounded 𝕜 (f '' V)} {x : E} :
    f.clmOfExistsBoundedImage h x = f x :=
  rfl


theorem LinearMap.continuousAt_zero_of_locally_bounded (f : E →ₛₗ[σ] F)
    (hf : ∀ s, IsVonNBounded 𝕜 s → IsVonNBounded 𝕜' (f '' s)) : ContinuousAt f 0 := by
  -- Assume that f is not continuous at 0
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹¹ : AddCommGroup E
    inst✝¹⁰ : UniformSpace E
    inst✝⁹ : UniformAddGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : UniformSpace F
    inst✝⁶ : FirstCountableTopology E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : ContinuousSMul 𝕜 E
    inst✝² : RCLike 𝕜'
    inst✝¹ : Module 𝕜' F
    inst✝ : ContinuousSMul 𝕜' F
    σ : RingHom 𝕜 𝕜'
    f : LinearMap σ E F
    hf : ∀ (s : Set E), Bornology.IsVonNBounded 𝕜 s → Bornology.IsVonNBounded 𝕜' ( …
    ⊢ ContinuousAt (⇑f) 0
  -/
  by_contra h
  -- We use a decreasing balanced basis for 0 : E and a balanced basis for 0 : F
  -- and reformulate non-continuity in terms of these bases
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹¹ : AddCommGroup E
    inst✝¹⁰ : UniformSpace E
    inst✝⁹ : UniformAddGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : UniformSpace F
    inst✝⁶ : FirstCountableTopology E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : ContinuousSMul 𝕜 E
    inst✝² : RCLike 𝕜'
    inst✝¹ : Module 𝕜' F
    inst✝ : ContinuousSMul 𝕜' F
    σ : RingHom 𝕜 𝕜'
    f : LinearMap σ E F
    hf : ∀ (s : Set E), Bornology.IsVonNBounded 𝕜 s → Bornology.IsVonNBounded 𝕜' ( …
    h : Not (ContinuousAt (⇑f) 0)
    ⊢ False
  -/
  rcases (nhds_basis_balanced 𝕜 E).exists_antitone_subbasis with ⟨b, bE1, bE⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹¹ : AddCommGroup E
    inst✝¹⁰ : UniformSpace E
    inst✝⁹ : UniformAddGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : UniformSpace F
    inst✝⁶ : FirstCountableTopology E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : ContinuousSMul 𝕜 E
    inst✝² : RCLike 𝕜'
    inst✝¹ : Module 𝕜' F
    inst✝ : ContinuousSMul 𝕜' F
    σ : RingHom 𝕜 𝕜'
    f : LinearMap σ E F
    hf : ∀ (s : Set E), Bornology.IsVonNBounded 𝕜 s → Bornology.IsVonNBounded 𝕜' ( …
    h : Not (ContinuousAt (⇑f) 0)
    b : Nat → Set E
    bE1 : ∀ (i : Nat), And (Membership.mem (nhds 0) (b i)) (Balanced 𝕜 (b i))
    bE : (nhds 0).HasAntitoneBasis fun i => _root_.id (b i)
    ⊢ False
  -/
  simp only [_root_.id] at bE
  have bE' : (𝓝 (0 : E)).HasBasis (fun x : ℕ => x ≠ 0) fun n : ℕ => (n : 𝕜)⁻¹ • b n := by
    refine bE.1.to_hasBasis ?_ ?_
    · intro n _
      use n + 1
      simp only [Ne, Nat.succ_ne_zero, not_false_iff, Nat.cast_add, Nat.cast_one, true_and]
      -- `b (n + 1) ⊆ b n` follows from `Antitone`.
      have h : b (n + 1) ⊆ b n := bE.2 (by simp)
      refine _root_.trans ?_ h
      rintro y ⟨x, hx, hy⟩
      -- Since `b (n + 1)` is balanced `(n+1)⁻¹ b (n + 1) ⊆ b (n + 1)`
      rw [← hy]
      refine (bE1 (n + 1)).2.smul_mem ?_ hx
      have h' : 0 < (n : ℝ) + 1 := n.cast_add_one_pos
      rw [norm_inv, ← Nat.cast_one, ← Nat.cast_add, RCLike.norm_natCast, Nat.cast_add,
        Nat.cast_one, inv_le_comm₀ h' zero_lt_one]
      simp
    intro n hn
    -- The converse direction follows from continuity of the scalar multiplication
    have hcont : ContinuousAt (fun x : E => (n : 𝕜) • x) 0 :=
      (continuous_const_smul (n : 𝕜)).continuousAt
    simp only [ContinuousAt, map_zero, smul_zero] at hcont
    rw [bE.1.tendsto_left_iff] at hcont
    rcases hcont (b n) (bE1 n).1 with ⟨i, _, hi⟩
    refine ⟨i, trivial, fun x hx => ⟨(n : 𝕜) • x, hi hx, ?_⟩⟩
    simp [← mul_smul, hn]
  /-
    case intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹¹ : AddCommGroup E
    inst✝¹⁰ : UniformSpace E
    inst✝⁹ : UniformAddGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : UniformSpace F
    inst✝⁶ : FirstCountableTopology E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : ContinuousSMul 𝕜 E
    inst✝² : RCLike 𝕜'
    inst✝¹ : Module 𝕜' F
    inst✝ : ContinuousSMul 𝕜' F
    σ : RingHom 𝕜 𝕜'
    f : LinearMap σ E F
    hf : ∀ (s : Set E), Bornology.IsVonNBounded 𝕜 s → Bornology.IsVonNBounded 𝕜' ( …
    h : Not (ContinuousAt (⇑f) 0)
    b : Nat → Set E
    bE1 : ∀ (i : Nat), And (Membership.mem (nhds 0) (b i)) (Balanced 𝕜 (b i))
    bE : (nhds 0).HasAntitoneBasis fun i => b i
    bE' : (nhds 0).HasBasis (fun x => Ne x 0) fun n => HSMul.hSMul (Inv.inv ↑n) (b …
    ⊢ False
  -/
  rw [ContinuousAt, map_zero, bE'.tendsto_iff (nhds_basis_balanced 𝕜' F)] at h
  /-
    case intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹¹ : AddCommGroup E
    inst✝¹⁰ : UniformSpace E
    inst✝⁹ : UniformAddGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : UniformSpace F
    inst✝⁶ : FirstCountableTopology E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : ContinuousSMul 𝕜 E
    inst✝² : RCLike 𝕜'
    inst✝¹ : Module 𝕜' F
    inst✝ : ContinuousSMul 𝕜' F
    σ : RingHom 𝕜 𝕜'
    f : LinearMap σ E F
    hf : ∀ (s : Set E), Bornology.IsVonNBounded 𝕜 s → Bornology.IsVonNBounded 𝕜' ( …
    b : Nat → Set E
    h : Not (∀ (ib : Set F), And (Membership.mem (nhds 0) ib) (Balanced 𝕜' ib) → E …
    bE1 : ∀ (i : Nat), And (Membership.mem (nhds 0) (b i)) (Balanced 𝕜 (b i))
    bE : (nhds 0).HasAntitoneBasis fun i => b i
    bE' : (nhds 0).HasBasis (fun x => Ne x 0) fun n => HSMul.hSMul (Inv.inv ↑n) (b …
    ⊢ False
  -/
  push_neg at h
  /-
    case intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹¹ : AddCommGroup E
    inst✝¹⁰ : UniformSpace E
    inst✝⁹ : UniformAddGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : UniformSpace F
    inst✝⁶ : FirstCountableTopology E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : ContinuousSMul 𝕜 E
    inst✝² : RCLike 𝕜'
    inst✝¹ : Module 𝕜' F
    inst✝ : ContinuousSMul 𝕜' F
    σ : RingHom 𝕜 𝕜'
    f : LinearMap σ E F
    hf : ∀ (s : Set E), Bornology.IsVonNBounded 𝕜 s → Bornology.IsVonNBounded 𝕜' ( …
    b : Nat → Set E
    bE1 : ∀ (i : Nat), And (Membership.mem (nhds 0) (b i)) (Balanced 𝕜 (b i))
    bE : (nhds 0).HasAntitoneBasis fun i => b i
    bE' : (nhds 0).HasBasis (fun x => Ne x 0) fun n => HSMul.hSMul (Inv.inv ↑n) (b …
    h : Exists fun ib => And (And (Membership.mem (nhds 0) ib) (Balanced 𝕜' ib)) ( …
    ⊢ False
  -/
  rcases h with ⟨V, ⟨hV, -⟩, h⟩
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹¹ : AddCommGroup E
    inst✝¹⁰ : UniformSpace E
    inst✝⁹ : UniformAddGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : UniformSpace F
    inst✝⁶ : FirstCountableTopology E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : ContinuousSMul 𝕜 E
    inst✝² : RCLike 𝕜'
    inst✝¹ : Module 𝕜' F
    inst✝ : ContinuousSMul 𝕜' F
    σ : RingHom 𝕜 𝕜'
    f : LinearMap σ E F
    hf : ∀ (s : Set E), Bornology.IsVonNBounded 𝕜 s → Bornology.IsVonNBounded 𝕜' ( …
    b : Nat → Set E
    bE1 : ∀ (i : Nat), And (Membership.mem (nhds 0) (b i)) (Balanced 𝕜 (b i))
    bE : (nhds 0).HasAntitoneBasis fun i => b i
    bE' : (nhds 0).HasBasis (fun x => Ne x 0) fun n => HSMul.hSMul (Inv.inv ↑n) (b …
    V : Set F
    h : ∀ (ia : Nat), Ne ia 0 → Exists fun x => And (Membership.mem (HSMul.hSMul ( …
    hV : Membership.mem (nhds 0) V
    ⊢ False
  -/
  simp only [_root_.id, forall_true_left] at h
  -- There exists `u : ℕ → E` such that for all `n : ℕ` we have `u n ∈ n⁻¹ • b n` and `f (u n) ∉ V`
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹¹ : AddCommGroup E
    inst✝¹⁰ : UniformSpace E
    inst✝⁹ : UniformAddGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : UniformSpace F
    inst✝⁶ : FirstCountableTopology E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : ContinuousSMul 𝕜 E
    inst✝² : RCLike 𝕜'
    inst✝¹ : Module 𝕜' F
    inst✝ : ContinuousSMul 𝕜' F
    σ : RingHom 𝕜 𝕜'
    f : LinearMap σ E F
    hf : ∀ (s : Set E), Bornology.IsVonNBounded 𝕜 s → Bornology.IsVonNBounded 𝕜' ( …
    b : Nat → Set E
    bE1 : ∀ (i : Nat), And (Membership.mem (nhds 0) (b i)) (Balanced 𝕜 (b i))
    bE : (nhds 0).HasAntitoneBasis fun i => b i
    bE' : (nhds 0).HasBasis (fun x => Ne x 0) fun n => HSMul.hSMul (Inv.inv ↑n) (b …
    V : Set F
    h : ∀ (ia : Nat), Ne ia 0 → Exists fun x => And (Membership.mem (HSMul.hSMul ( …
    hV : Membership.mem (nhds 0) V
    ⊢ False
  -/
  choose! u hu hu' using h
  -- The sequence `(fun n ↦ n • u n)` converges to `0`
  have h_tendsto : Tendsto (fun n : ℕ => (n : 𝕜) • u n) atTop (𝓝 (0 : E)) := by
    apply bE.tendsto
    intro n
    by_cases h : n = 0
    · rw [h, Nat.cast_zero, zero_smul]
      exact mem_of_mem_nhds (bE.1.mem_of_mem <| by trivial)
    rcases hu n h with ⟨y, hy, hu1⟩
    convert hy
    rw [← hu1, ← mul_smul]
    simp only [h, mul_inv_cancel₀, Ne, Nat.cast_eq_zero, not_false_iff, one_smul]
  -- The image `(fun n ↦ n • u n)` is von Neumann bounded:
  have h_bounded : IsVonNBounded 𝕜 (Set.range fun n : ℕ => (n : 𝕜) • u n) :=
    h_tendsto.cauchySeq.totallyBounded_range.isVonNBounded 𝕜
  -- Since `range u` is bounded, `V` absorbs it
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹¹ : AddCommGroup E
    inst✝¹⁰ : UniformSpace E
    inst✝⁹ : UniformAddGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : UniformSpace F
    inst✝⁶ : FirstCountableTopology E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : ContinuousSMul 𝕜 E
    inst✝² : RCLike 𝕜'
    inst✝¹ : Module 𝕜' F
    inst✝ : ContinuousSMul 𝕜' F
    σ : RingHom 𝕜 𝕜'
    f : LinearMap σ E F
    hf : ∀ (s : Set E), Bornology.IsVonNBounded 𝕜 s → Bornology.IsVonNBounded 𝕜' ( …
    b : Nat → Set E
    bE1 : ∀ (i : Nat), And (Membership.mem (nhds 0) (b i)) (Balanced 𝕜 (b i))
    bE : (nhds 0).HasAntitoneBasis fun i => b i
    bE' : (nhds 0).HasBasis (fun x => Ne x 0) fun n => HSMul.hSMul (Inv.inv ↑n) (b …
    V : Set F
    hV : Membership.mem (nhds 0) V
    u : Nat → E
    hu : ∀ (ia : Nat), Ne ia 0 → Membership.mem (HSMul.hSMul (Inv.inv ↑ia) (b ia)) …
    hu' : ∀ (ia : Nat), Ne ia 0 → Not (Membership.mem V (f (u ia)))
    h_tendsto : Filter.Tendsto (fun n => HSMul.hSMul (↑n) (u n)) Filter.atTop (nhd …
    h_bounded : Bornology.IsVonNBounded 𝕜 (Set.range fun n => HSMul.hSMul (↑n) (u  …
    ⊢ False
  -/
  rcases (hf _ h_bounded hV).exists_pos with ⟨r, hr, h'⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹¹ : AddCommGroup E
    inst✝¹⁰ : UniformSpace E
    inst✝⁹ : UniformAddGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : UniformSpace F
    inst✝⁶ : FirstCountableTopology E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : ContinuousSMul 𝕜 E
    inst✝² : RCLike 𝕜'
    inst✝¹ : Module 𝕜' F
    inst✝ : ContinuousSMul 𝕜' F
    σ : RingHom 𝕜 𝕜'
    f : LinearMap σ E F
    hf : ∀ (s : Set E), Bornology.IsVonNBounded 𝕜 s → Bornology.IsVonNBounded 𝕜' ( …
    b : Nat → Set E
    bE1 : ∀ (i : Nat), And (Membership.mem (nhds 0) (b i)) (Balanced 𝕜 (b i))
    bE : (nhds 0).HasAntitoneBasis fun i => b i
    bE' : (nhds 0).HasBasis (fun x => Ne x 0) fun n => HSMul.hSMul (Inv.inv ↑n) (b …
    V : Set F
    hV : Membership.mem (nhds 0) V
    u : Nat → E
    hu : ∀ (ia : Nat), Ne ia 0 → Membership.mem (HSMul.hSMul (Inv.inv ↑ia) (b ia)) …
    hu' : ∀ (ia : Nat), Ne ia 0 → Not (Membership.mem V (f (u ia)))
    h_tendsto : Filter.Tendsto (fun n => HSMul.hSMul (↑n) (u n)) Filter.atTop (nhd …
    h_bounded : Bornology.IsVonNBounded 𝕜 (Set.range fun n => HSMul.hSMul (↑n) (u  …
    r : Real
    hr : GT.gt r 0
    h' : ∀ (c : 𝕜'), LE.le r (Norm.norm c) → HasSubset.Subset (Set.image (⇑f) (Set …
    ⊢ False
  -/
  cases' exists_nat_gt r with n hn
  -- We now find a contradiction between `f (u n) ∉ V` and the absorbing property
  have h1 : r ≤ ‖(n : 𝕜')‖ := by
    rw [RCLike.norm_natCast]
    exact hn.le
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹¹ : AddCommGroup E
    inst✝¹⁰ : UniformSpace E
    inst✝⁹ : UniformAddGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : UniformSpace F
    inst✝⁶ : FirstCountableTopology E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : ContinuousSMul 𝕜 E
    inst✝² : RCLike 𝕜'
    inst✝¹ : Module 𝕜' F
    inst✝ : ContinuousSMul 𝕜' F
    σ : RingHom 𝕜 𝕜'
    f : LinearMap σ E F
    hf : ∀ (s : Set E), Bornology.IsVonNBounded 𝕜 s → Bornology.IsVonNBounded 𝕜' ( …
    b : Nat → Set E
    bE1 : ∀ (i : Nat), And (Membership.mem (nhds 0) (b i)) (Balanced 𝕜 (b i))
    bE : (nhds 0).HasAntitoneBasis fun i => b i
    bE' : (nhds 0).HasBasis (fun x => Ne x 0) fun n => HSMul.hSMul (Inv.inv ↑n) (b …
    V : Set F
    hV : Membership.mem (nhds 0) V
    u : Nat → E
    hu : ∀ (ia : Nat), Ne ia 0 → Membership.mem (HSMul.hSMul (Inv.inv ↑ia) (b ia)) …
    hu' : ∀ (ia : Nat), Ne ia 0 → Not (Membership.mem V (f (u ia)))
    h_tendsto : Filter.Tendsto (fun n => HSMul.hSMul (↑n) (u n)) Filter.atTop (nhd …
    h_bounded : Bornology.IsVonNBounded 𝕜 (Set.range fun n => HSMul.hSMul (↑n) (u  …
    r : Real
    hr : GT.gt r 0
    h' : ∀ (c : 𝕜'), LE.le r (Norm.norm c) → HasSubset.Subset (Set.image (⇑f) (Set …
    n : Nat
    hn : LT.lt r ↑n
    h1 : LE.le r (Norm.norm ↑n)
    ⊢ False
  -/
  have hn' : 0 < ‖(n : 𝕜')‖ := lt_of_lt_of_le hr h1
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹¹ : AddCommGroup E
    inst✝¹⁰ : UniformSpace E
    inst✝⁹ : UniformAddGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : UniformSpace F
    inst✝⁶ : FirstCountableTopology E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : ContinuousSMul 𝕜 E
    inst✝² : RCLike 𝕜'
    inst✝¹ : Module 𝕜' F
    inst✝ : ContinuousSMul 𝕜' F
    σ : RingHom 𝕜 𝕜'
    f : LinearMap σ E F
    hf : ∀ (s : Set E), Bornology.IsVonNBounded 𝕜 s → Bornology.IsVonNBounded 𝕜' ( …
    b : Nat → Set E
    bE1 : ∀ (i : Nat), And (Membership.mem (nhds 0) (b i)) (Balanced 𝕜 (b i))
    bE : (nhds 0).HasAntitoneBasis fun i => b i
    bE' : (nhds 0).HasBasis (fun x => Ne x 0) fun n => HSMul.hSMul (Inv.inv ↑n) (b …
    V : Set F
    hV : Membership.mem (nhds 0) V
    u : Nat → E
    hu : ∀ (ia : Nat), Ne ia 0 → Membership.mem (HSMul.hSMul (Inv.inv ↑ia) (b ia)) …
    hu' : ∀ (ia : Nat), Ne ia 0 → Not (Membership.mem V (f (u ia)))
    h_tendsto : Filter.Tendsto (fun n => HSMul.hSMul (↑n) (u n)) Filter.atTop (nhd …
    h_bounded : Bornology.IsVonNBounded 𝕜 (Set.range fun n => HSMul.hSMul (↑n) (u  …
    r : Real
    hr : GT.gt r 0
    h' : ∀ (c : 𝕜'), LE.le r (Norm.norm c) → HasSubset.Subset (Set.image (⇑f) (Set …
    n : Nat
    hn : LT.lt r ↑n
    h1 : LE.le r (Norm.norm ↑n)
    hn' : LT.lt 0 (Norm.norm ↑n)
    ⊢ False
  -/
  rw [norm_pos_iff, Ne, Nat.cast_eq_zero] at hn'
  have h'' : f (u n) ∈ V := by
    simp only [Set.image_subset_iff] at h'
    specialize h' (n : 𝕜') h1 (Set.mem_range_self n)
    simp only [Set.mem_preimage, LinearMap.map_smulₛₗ, map_natCast] at h'
    rcases h' with ⟨y, hy, h'⟩
    apply_fun fun y : F => (n : 𝕜')⁻¹ • y at h'
    simp only [hn', inv_smul_smul₀, Ne, Nat.cast_eq_zero, not_false_iff] at h'
    rwa [← h']
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹¹ : AddCommGroup E
    inst✝¹⁰ : UniformSpace E
    inst✝⁹ : UniformAddGroup E
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : UniformSpace F
    inst✝⁶ : FirstCountableTopology E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : ContinuousSMul 𝕜 E
    inst✝² : RCLike 𝕜'
    inst✝¹ : Module 𝕜' F
    inst✝ : ContinuousSMul 𝕜' F
    σ : RingHom 𝕜 𝕜'
    f : LinearMap σ E F
    hf : ∀ (s : Set E), Bornology.IsVonNBounded 𝕜 s → Bornology.IsVonNBounded 𝕜' ( …
    b : Nat → Set E
    bE1 : ∀ (i : Nat), And (Membership.mem (nhds 0) (b i)) (Balanced 𝕜 (b i))
    bE : (nhds 0).HasAntitoneBasis fun i => b i
    bE' : (nhds 0).HasBasis (fun x => Ne x 0) fun n => HSMul.hSMul (Inv.inv ↑n) (b …
    V : Set F
    hV : Membership.mem (nhds 0) V
    u : Nat → E
    hu : ∀ (ia : Nat), Ne ia 0 → Membership.mem (HSMul.hSMul (Inv.inv ↑ia) (b ia)) …
    hu' : ∀ (ia : Nat), Ne ia 0 → Not (Membership.mem V (f (u ia)))
    h_tendsto : Filter.Tendsto (fun n => HSMul.hSMul (↑n) (u n)) Filter.atTop (nhd …
    h_bounded : Bornology.IsVonNBounded 𝕜 (Set.range fun n => HSMul.hSMul (↑n) (u  …
    r : Real
    hr : GT.gt r 0
    h' : ∀ (c : 𝕜'), LE.le r (Norm.norm c) → HasSubset.Subset (Set.image (⇑f) (Set …
    n : Nat
    hn : LT.lt r ↑n
    h1 : LE.le r (Norm.norm ↑n)
    hn' : Not (Eq n 0)
    h'' : Membership.mem V (f (u n))
    ⊢ False
  -/
  exact hu' n hn' h''
  /-
    🎉 no goals
  -/


/-- If `E` is first countable, then every locally bounded linear map `E →ₛₗ[σ] F` is continuous. -/
theorem LinearMap.continuous_of_locally_bounded [UniformAddGroup F] (f : E →ₛₗ[σ] F)
    (hf : ∀ s, IsVonNBounded 𝕜 s → IsVonNBounded 𝕜' (f '' s)) : Continuous f :=
  (uniformContinuous_of_continuousAt_zero f <| f.continuousAt_zero_of_locally_bounded hf).continuous



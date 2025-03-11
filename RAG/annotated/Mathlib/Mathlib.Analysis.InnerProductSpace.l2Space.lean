local notation "⟪" x ", " y "⟫" => @inner 𝕜 _ _ x y


/-- `ℓ²(ι, 𝕜)` is the Hilbert space of square-summable functions `ι → 𝕜`, herein implemented
as `lp (fun i : ι => 𝕜) 2`. -/
notation "ℓ²(" ι ", " 𝕜 ")" => lp (fun i : ι => 𝕜) 2


theorem summable_inner (f g : lp G 2) : Summable fun i => ⟪f i, g i⟫ := by
  -- Apply the Direct Comparison Test, comparing with ∑' i, ‖f i‖ * ‖g i‖ (summable by Hölder)
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    G : ι → Type u_4
    inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
    inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
    f g : Subtype fun x => Membership.mem (lp G 2) x
    ⊢ Summable fun i => Inner.inner (↑f i) (↑g i)
  -/
  refine .of_norm_bounded (fun i => ‖f i‖ * ‖g i‖) (lp.summable_mul ?_ f g) ?_
    /-
      case refine_1
      ι : Type u_1
      𝕜 : Type u_2
      inst✝² : RCLike 𝕜
      G : ι → Type u_4
      inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
      inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
      f g : Subtype fun x => Membership.mem (lp G 2) x
      ⊢ (ENNReal.toReal 2).IsConjExponent (ENNReal.toReal 2)
    -/
  · rw [Real.isConjExponent_iff]; norm_num
                                  /-
                                    🎉 no goals
                                  -/
  /-
    case refine_2
    ι : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    G : ι → Type u_4
    inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
    inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
    f g : Subtype fun x => Membership.mem (lp G 2) x
    ⊢ ∀ (i : ι), LE.le (Norm.norm (Inner.inner (↑f i) (↑g i))) ((fun i => HMul.hMu …
  -/
  intro i
  -- Then apply Cauchy-Schwarz pointwise
  /-
    case refine_2
    ι : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    G : ι → Type u_4
    inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
    inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
    f g : Subtype fun x => Membership.mem (lp G 2) x
    i : ι
    ⊢ LE.le (Norm.norm (Inner.inner (↑f i) (↑g i))) ((fun i => HMul.hMul (Norm.nor …
  -/
  exact norm_inner_le_norm (𝕜 := 𝕜) _ _
  /-
    🎉 no goals
  -/


instance instInnerProductSpace : InnerProductSpace 𝕜 (lp G 2) :=
  { lp.normedAddCommGroup (E := G) (p := 2) with
    inner := fun f g => ∑' i, ⟪f i, g i⟫
    norm_sq_eq_inner := fun f => by
      calc
        ‖f‖ ^ 2 = ‖f‖ ^ (2 : ℝ≥0∞).toReal := by norm_cast
        _ = ∑' i, ‖f i‖ ^ (2 : ℝ≥0∞).toReal := lp.norm_rpow_eq_tsum ?_ f
        _ = ∑' i, ‖f i‖ ^ (2 : ℕ) := by norm_cast
        _ = ∑' i, re ⟪f i, f i⟫ := by
          congr
          funext i
          rw [norm_sq_eq_inner (𝕜 := 𝕜)]
          -- Porting note: `simp` couldn't do this anymore
        _ = re (∑' i, ⟪f i, f i⟫) := (RCLike.reCLM.map_tsum ?_).symm
        /-
          case calc_1
          ι : Type u_1
          𝕜 : Type u_2
          inst✝⁴ : RCLike 𝕜
          E : Type u_3
          inst✝³ : NormedAddCommGroup E
          inst✝² : InnerProductSpace 𝕜 E
          G : ι → Type u_4
          inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
          inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
          f : Subtype fun x => Membership.mem (lp G 2) x
          ⊢ LT.lt 0 (ENNReal.toReal 2)
        -/
      · norm_num
        /-
          🎉 no goals
        -/
        /-
          case calc_2
          ι : Type u_1
          𝕜 : Type u_2
          inst✝⁴ : RCLike 𝕜
          E : Type u_3
          inst✝³ : NormedAddCommGroup E
          inst✝² : InnerProductSpace 𝕜 E
          G : ι → Type u_4
          inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
          inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
          f : Subtype fun x => Membership.mem (lp G 2) x
          ⊢ Summable fun i => Inner.inner (↑f i) (↑f i)
        -/
      · exact summable_inner f f
        /-
          🎉 no goals
        -/
    conj_symm := fun f g => by
      calc
        conj _ = conj (∑' i, ⟪g i, f i⟫) := by congr
        _ = ∑' i, conj ⟪g i, f i⟫ := RCLike.conjCLE.map_tsum
        _ = ∑' i, ⟪f i, g i⟫ := by simp only [inner_conj_symm]
        _ = _ := by congr
    add_left := fun f₁ f₂ g => by
      calc
        _ = ∑' i, ⟪(f₁ + f₂) i, g i⟫ := ?_
        _ = ∑' i, (⟪f₁ i, g i⟫ + ⟪f₂ i, g i⟫) := by
          simp only [inner_add_left, Pi.add_apply, coeFn_add]
        _ = (∑' i, ⟪f₁ i, g i⟫) + ∑' i, ⟪f₂ i, g i⟫ := tsum_add ?_ ?_
        _ = _ := by congr
        /-
          case calc_1
          ι : Type u_1
          𝕜 : Type u_2
          inst✝⁴ : RCLike 𝕜
          E : Type u_3
          inst✝³ : NormedAddCommGroup E
          inst✝² : InnerProductSpace 𝕜 E
          G : ι → Type u_4
          inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
          inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
          f₁ f₂ g : Subtype fun x => Membership.mem (lp G 2) x
          ⊢ Eq (Inner.inner (HAdd.hAdd f₁ f₂) g) (tsum fun i => Inner.inner (↑(HAdd.hAdd …
        -/
      · congr
        /-
          🎉 no goals
        -/
        /-
          case calc_2
          ι : Type u_1
          𝕜 : Type u_2
          inst✝⁴ : RCLike 𝕜
          E : Type u_3
          inst✝³ : NormedAddCommGroup E
          inst✝² : InnerProductSpace 𝕜 E
          G : ι → Type u_4
          inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
          inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
          f₁ f₂ g : Subtype fun x => Membership.mem (lp G 2) x
          ⊢ Summable fun i => Inner.inner (↑f₁ i) (↑g i)
        -/
      · exact summable_inner f₁ g
        /-
          🎉 no goals
        -/
        /-
          case calc_3
          ι : Type u_1
          𝕜 : Type u_2
          inst✝⁴ : RCLike 𝕜
          E : Type u_3
          inst✝³ : NormedAddCommGroup E
          inst✝² : InnerProductSpace 𝕜 E
          G : ι → Type u_4
          inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
          inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
          f₁ f₂ g : Subtype fun x => Membership.mem (lp G 2) x
          ⊢ Summable fun i => Inner.inner (↑f₂ i) (↑g i)
        -/
      · exact summable_inner f₂ g
        /-
          🎉 no goals
        -/
    smul_left := fun f g c => by
      calc
        _ = ∑' i, ⟪c • f i, g i⟫ := ?_
        _ = ∑' i, conj c * ⟪f i, g i⟫ := by simp only [inner_smul_left]
        _ = conj c * ∑' i, ⟪f i, g i⟫ := tsum_mul_left
        _ = _ := ?_
        /-
          case calc_1
          ι : Type u_1
          𝕜 : Type u_2
          inst✝⁴ : RCLike 𝕜
          E : Type u_3
          inst✝³ : NormedAddCommGroup E
          inst✝² : InnerProductSpace 𝕜 E
          G : ι → Type u_4
          inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
          inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
          f g : Subtype fun x => Membership.mem (lp G 2) x
          c : 𝕜
          ⊢ Eq (Inner.inner (HSMul.hSMul c f) g) (tsum fun i => Inner.inner (HSMul.hSMul …
        -/
      · simp only [coeFn_smul, Pi.smul_apply]
        /-
          🎉 no goals
        -/
        /-
          case calc_2
          ι : Type u_1
          𝕜 : Type u_2
          inst✝⁴ : RCLike 𝕜
          E : Type u_3
          inst✝³ : NormedAddCommGroup E
          inst✝² : InnerProductSpace 𝕜 E
          G : ι → Type u_4
          inst✝¹ : (i : ι) → NormedAddCommGroup (G i)
          inst✝ : (i : ι) → InnerProductSpace 𝕜 (G i)
          f g : Subtype fun x => Membership.mem (lp G 2) x
          c : 𝕜
          ⊢ Eq (HMul.hMul ((starRingEnd 𝕜) c) (tsum fun i => Inner.inner (↑f i) (↑g i))) …
        -/
      · congr }
        /-
          🎉 no goals
        -/


theorem inner_eq_tsum (f g : lp G 2) : ⟪f, g⟫ = ∑' i, ⟪f i, g i⟫ :=
  rfl


theorem hasSum_inner (f g : lp G 2) : HasSum (fun i => ⟪f i, g i⟫) ⟪f, g⟫ :=
  (summable_inner f g).hasSum


theorem inner_single_left [DecidableEq ι] (i : ι) (a : G i) (f : lp G 2) :
    ⟪lp.single 2 i a, f⟫ = ⟪a, f i⟫ := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    G : ι → Type u_4
    inst✝² : (i : ι) → NormedAddCommGroup (G i)
    inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝ : DecidableEq ι
    i : ι
    a : G i
    f : Subtype fun x => Membership.mem (lp G 2) x
    ⊢ Eq (Inner.inner (lp.single 2 i a) f) (Inner.inner a (↑f i))
  -/
  refine (hasSum_inner (lp.single 2 i a) f).unique ?_
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    G : ι → Type u_4
    inst✝² : (i : ι) → NormedAddCommGroup (G i)
    inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝ : DecidableEq ι
    i : ι
    a : G i
    f : Subtype fun x => Membership.mem (lp G 2) x
    ⊢ HasSum (fun i_1 => Inner.inner (↑(lp.single 2 i a) i_1) (↑f i_1)) (Inner.inn …
  -/
  convert hasSum_ite_eq i ⟪a, f i⟫ using 1
  /-
    case h.e'_5
    ι : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    G : ι → Type u_4
    inst✝² : (i : ι) → NormedAddCommGroup (G i)
    inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝ : DecidableEq ι
    i : ι
    a : G i
    f : Subtype fun x => Membership.mem (lp G 2) x
    ⊢ Eq (fun i_1 => Inner.inner (↑(lp.single 2 i a) i_1) (↑f i_1)) fun b' => ite  …
  -/
  ext j
  /-
    case h.e'_5.h
    ι : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    G : ι → Type u_4
    inst✝² : (i : ι) → NormedAddCommGroup (G i)
    inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝ : DecidableEq ι
    i : ι
    a : G i
    f : Subtype fun x => Membership.mem (lp G 2) x
    j : ι
    ⊢ Eq (Inner.inner (↑(lp.single 2 i a) j) (↑f j)) (ite (Eq j i) (Inner.inner a  …
  -/
  rw [lp.single_apply]
  /-
    case h.e'_5.h
    ι : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    G : ι → Type u_4
    inst✝² : (i : ι) → NormedAddCommGroup (G i)
    inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝ : DecidableEq ι
    i : ι
    a : G i
    f : Subtype fun x => Membership.mem (lp G 2) x
    j : ι
    ⊢ Eq (Inner.inner (dite (Eq j i) (fun h => Eq.ndrec a ⋯) fun h => 0) (↑f j)) ( …
  -/
  split_ifs with h
    /-
      case pos
      ι : Type u_1
      𝕜 : Type u_2
      inst✝³ : RCLike 𝕜
      G : ι → Type u_4
      inst✝² : (i : ι) → NormedAddCommGroup (G i)
      inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
      inst✝ : DecidableEq ι
      i : ι
      a : G i
      f : Subtype fun x => Membership.mem (lp G 2) x
      j : ι
      h : Eq j i
      ⊢ Eq (Inner.inner (Eq.ndrec a ⋯) (↑f j)) (Inner.inner a (↑f i))
    -/
  · subst h; rfl
             /-
               🎉 no goals
             -/
    /-
      case neg
      ι : Type u_1
      𝕜 : Type u_2
      inst✝³ : RCLike 𝕜
      G : ι → Type u_4
      inst✝² : (i : ι) → NormedAddCommGroup (G i)
      inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
      inst✝ : DecidableEq ι
      i : ι
      a : G i
      f : Subtype fun x => Membership.mem (lp G 2) x
      j : ι
      h : Not (Eq j i)
      ⊢ Eq (Inner.inner 0 (↑f j)) 0
    -/
  · simp
    /-
      🎉 no goals
    -/


theorem inner_single_right [DecidableEq ι] (i : ι) (a : G i) (f : lp G 2) :
    ⟪f, lp.single 2 i a⟫ = ⟪f i, a⟫ := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    G : ι → Type u_4
    inst✝² : (i : ι) → NormedAddCommGroup (G i)
    inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝ : DecidableEq ι
    i : ι
    a : G i
    f : Subtype fun x => Membership.mem (lp G 2) x
    ⊢ Eq (Inner.inner f (lp.single 2 i a)) (Inner.inner (↑f i) a)
  -/
  simpa [inner_conj_symm] using congr_arg conj (inner_single_left (𝕜 := 𝕜) i a f)
  /-
    🎉 no goals
  -/


protected theorem summable_of_lp (f : lp G 2) :
    Summable fun i => V i (f i) := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝⁵ : RCLike 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    G : ι → Type u_4
    inst✝² : (i : ι) → NormedAddCommGroup (G i)
    inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝ : CompleteSpace E
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : OrthogonalFamily 𝕜 G V
    f : Subtype fun x => Membership.mem (lp G 2) x
    ⊢ Summable fun i => (V i) (↑f i)
  -/
  rw [hV.summable_iff_norm_sq_summable]
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝⁵ : RCLike 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    G : ι → Type u_4
    inst✝² : (i : ι) → NormedAddCommGroup (G i)
    inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝ : CompleteSpace E
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : OrthogonalFamily 𝕜 G V
    f : Subtype fun x => Membership.mem (lp G 2) x
    ⊢ Summable fun i => HPow.hPow (Norm.norm (↑f i)) 2
  -/
  convert (lp.memℓp f).summable _
    /-
      case h.e'_5.h
      ι : Type u_1
      𝕜 : Type u_2
      inst✝⁵ : RCLike 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      G : ι → Type u_4
      inst✝² : (i : ι) → NormedAddCommGroup (G i)
      inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
      inst✝ : CompleteSpace E
      V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
      hV : OrthogonalFamily 𝕜 G V
      f : Subtype fun x => Membership.mem (lp G 2) x
      x✝ : ι
      ⊢ Eq (HPow.hPow (Norm.norm (↑f x✝)) 2) (HPow.hPow (Norm.norm (↑f x✝)) (ENNReal …
    -/
  · norm_cast
    /-
      🎉 no goals
    -/
    /-
      ι : Type u_1
      𝕜 : Type u_2
      inst✝⁵ : RCLike 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : InnerProductSpace 𝕜 E
      G : ι → Type u_4
      inst✝² : (i : ι) → NormedAddCommGroup (G i)
      inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
      inst✝ : CompleteSpace E
      V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
      hV : OrthogonalFamily 𝕜 G V
      f : Subtype fun x => Membership.mem (lp G 2) x
      ⊢ LT.lt 0 (ENNReal.toReal 2)
    -/
  · norm_num
    /-
      🎉 no goals
    -/


/-- A mutually orthogonal family of subspaces of `E` induce a linear isometry from `lp 2` of the
subspaces into `E`. -/
protected def linearIsometry (hV : OrthogonalFamily 𝕜 G V) : lp G 2 →ₗᵢ[𝕜] E where
  toFun f := ∑' i, V i (f i)
  map_add' f g := by
    simp only [tsum_add (hV.summable_of_lp f) (hV.summable_of_lp g), lp.coeFn_add, Pi.add_apply,
      LinearIsometry.map_add]
  map_smul' c f := by
    simpa only [LinearIsometry.map_smul, Pi.smul_apply, lp.coeFn_smul] using
      tsum_const_smul c (hV.summable_of_lp f)
  norm_map' f := by
    classical
      -- needed for lattice instance on `Finset ι`, for `Filter.atTop_neBot`
      have H : 0 < (2 : ℝ≥0∞).toReal := by norm_num
      suffices ‖∑' i : ι, V i (f i)‖ ^ (2 : ℝ≥0∞).toReal = ‖f‖ ^ (2 : ℝ≥0∞).toReal by
        exact Real.rpow_left_injOn H.ne' (norm_nonneg _) (norm_nonneg _) this
      refine tendsto_nhds_unique ?_ (lp.hasSum_norm H f)
      convert (hV.summable_of_lp f).hasSum.norm.rpow_const (Or.inr H.le) using 1
      ext s
      exact mod_cast (hV.norm_sum f s).symm


protected theorem linearIsometry_apply (f : lp G 2) : hV.linearIsometry f = ∑' i, V i (f i) :=
  rfl


protected theorem hasSum_linearIsometry (f : lp G 2) :
    HasSum (fun i => V i (f i)) (hV.linearIsometry f) :=
  (hV.summable_of_lp f).hasSum


@[simp]
protected theorem linearIsometry_apply_single [DecidableEq ι] {i : ι} (x : G i) :
    hV.linearIsometry (lp.single 2 i x) = V i x := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝⁶ : RCLike 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace 𝕜 E
    G : ι → Type u_4
    inst✝³ : (i : ι) → NormedAddCommGroup (G i)
    inst✝² : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝¹ : CompleteSpace E
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : OrthogonalFamily 𝕜 G V
    inst✝ : DecidableEq ι
    i : ι
    x : G i
    ⊢ Eq (hV.linearIsometry (lp.single 2 i x)) ((V i) x)
  -/
  rw [hV.linearIsometry_apply, ← tsum_ite_eq i (V i x)]
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝⁶ : RCLike 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace 𝕜 E
    G : ι → Type u_4
    inst✝³ : (i : ι) → NormedAddCommGroup (G i)
    inst✝² : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝¹ : CompleteSpace E
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : OrthogonalFamily 𝕜 G V
    inst✝ : DecidableEq ι
    i : ι
    x : G i
    ⊢ Eq (tsum fun i_1 => (V i_1) (↑(lp.single 2 i x) i_1)) (tsum fun b' => ite (E …
  -/
  congr
  /-
    case e_f
    ι : Type u_1
    𝕜 : Type u_2
    inst✝⁶ : RCLike 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace 𝕜 E
    G : ι → Type u_4
    inst✝³ : (i : ι) → NormedAddCommGroup (G i)
    inst✝² : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝¹ : CompleteSpace E
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : OrthogonalFamily 𝕜 G V
    inst✝ : DecidableEq ι
    i : ι
    x : G i
    ⊢ Eq (fun i_1 => (V i_1) (↑(lp.single 2 i x) i_1)) fun b' => ite (Eq b' i) ((V …
  -/
  ext j
  /-
    case e_f.h
    ι : Type u_1
    𝕜 : Type u_2
    inst✝⁶ : RCLike 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace 𝕜 E
    G : ι → Type u_4
    inst✝³ : (i : ι) → NormedAddCommGroup (G i)
    inst✝² : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝¹ : CompleteSpace E
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : OrthogonalFamily 𝕜 G V
    inst✝ : DecidableEq ι
    i : ι
    x : G i
    j : ι
    ⊢ Eq ((V j) (↑(lp.single 2 i x) j)) (ite (Eq j i) ((V i) x) 0)
  -/
  rw [lp.single_apply]
  /-
    case e_f.h
    ι : Type u_1
    𝕜 : Type u_2
    inst✝⁶ : RCLike 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace 𝕜 E
    G : ι → Type u_4
    inst✝³ : (i : ι) → NormedAddCommGroup (G i)
    inst✝² : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝¹ : CompleteSpace E
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : OrthogonalFamily 𝕜 G V
    inst✝ : DecidableEq ι
    i : ι
    x : G i
    j : ι
    ⊢ Eq ((V j) (dite (Eq j i) (fun h => Eq.ndrec x ⋯) fun h => 0)) (ite (Eq j i)  …
  -/
  split_ifs with h
    /-
      case pos
      ι : Type u_1
      𝕜 : Type u_2
      inst✝⁶ : RCLike 𝕜
      E : Type u_3
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : InnerProductSpace 𝕜 E
      G : ι → Type u_4
      inst✝³ : (i : ι) → NormedAddCommGroup (G i)
      inst✝² : (i : ι) → InnerProductSpace 𝕜 (G i)
      inst✝¹ : CompleteSpace E
      V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
      hV : OrthogonalFamily 𝕜 G V
      inst✝ : DecidableEq ι
      i : ι
      x : G i
      j : ι
      h : Eq j i
      ⊢ Eq ((V j) (Eq.ndrec x ⋯)) ((V i) x)
    -/
  · subst h; simp
             /-
               🎉 no goals
             -/
    /-
      case neg
      ι : Type u_1
      𝕜 : Type u_2
      inst✝⁶ : RCLike 𝕜
      E : Type u_3
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : InnerProductSpace 𝕜 E
      G : ι → Type u_4
      inst✝³ : (i : ι) → NormedAddCommGroup (G i)
      inst✝² : (i : ι) → InnerProductSpace 𝕜 (G i)
      inst✝¹ : CompleteSpace E
      V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
      hV : OrthogonalFamily 𝕜 G V
      inst✝ : DecidableEq ι
      i : ι
      x : G i
      j : ι
      h : Not (Eq j i)
      ⊢ Eq ((V j) 0) 0
    -/
  · simp [h]
    /-
      🎉 no goals
    -/


protected theorem linearIsometry_apply_dfinsupp_sum_single [DecidableEq ι] [∀ i, DecidableEq (G i)]
    (W₀ : Π₀ i : ι, G i) : hV.linearIsometry (W₀.sum (lp.single 2)) = W₀.sum fun i => V i := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝⁷ : RCLike 𝕜
    E : Type u_3
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : InnerProductSpace 𝕜 E
    G : ι → Type u_4
    inst✝⁴ : (i : ι) → NormedAddCommGroup (G i)
    inst✝³ : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝² : CompleteSpace E
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : OrthogonalFamily 𝕜 G V
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    W₀ : DFinsupp fun i => G i
    ⊢ Eq (hV.linearIsometry (W₀.sum (lp.single 2))) (W₀.sum fun i => ⇑(V i))
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The canonical linear isometry from the `lp 2` of a mutually orthogonal family of subspaces of
`E` into E, has range the closure of the span of the subspaces. -/
protected theorem range_linearIsometry [∀ i, CompleteSpace (G i)] :
    LinearMap.range hV.linearIsometry.toLinearMap =
      (⨆ i, LinearMap.range (V i).toLinearMap).topologicalClosure := by
    -- Porting note: dot notation broken
  classical
  refine le_antisymm ?_ ?_
  · rintro x ⟨f, rfl⟩
    refine mem_closure_of_tendsto (hV.hasSum_linearIsometry f) (Eventually.of_forall ?_)
    intro s
    rw [SetLike.mem_coe]
    refine sum_mem ?_
    intro i _
    refine mem_iSup_of_mem i ?_
    exact LinearMap.mem_range_self _ (f i)
  · apply topologicalClosure_minimal
    · refine iSup_le ?_
      rintro i x ⟨x, rfl⟩
      use lp.single 2 i x
      exact hV.linearIsometry_apply_single x
    exact hV.linearIsometry.isometry.isUniformInducing.isComplete_range.isClosed


/-- Given a family of Hilbert spaces `G : ι → Type*`, a Hilbert sum of `G` consists of a Hilbert
space `E` and an orthogonal family `V : Π i, G i →ₗᵢ[𝕜] E` such that the induced isometry
`Φ : lp G 2 → E` is surjective.

Keeping in mind that `lp G 2` is "the" external Hilbert sum of `G : ι → Type*`, this is analogous
to `DirectSum.IsInternal`, except that we don't express it in terms of actual submodules. -/
structure IsHilbertSum : Prop where
  ofSurjective ::
  /-- The orthogonal family constituting the summands in the Hilbert sum. -/
  protected OrthogonalFamily : OrthogonalFamily 𝕜 G V
  /-- The isometry `lp G 2 → E` induced by the orthogonal family is surjective. -/
  protected surjective_isometry : Function.Surjective OrthogonalFamily.linearIsometry


/-- If `V : Π i, G i →ₗᵢ[𝕜] E` is an orthogonal family such that the supremum of the ranges of
`V i` is dense, then `(E, V)` is a Hilbert sum of `G`. -/
theorem IsHilbertSum.mk [∀ i, CompleteSpace <| G i] (hVortho : OrthogonalFamily 𝕜 G V)
    (hVtotal : ⊤ ≤ (⨆ i, LinearMap.range (V i).toLinearMap).topologicalClosure) :
    IsHilbertSum 𝕜 G V :=
  { OrthogonalFamily := hVortho
    surjective_isometry := by
      /-
        ι : Type u_1
        𝕜 : Type u_2
        inst✝⁶ : RCLike 𝕜
        E : Type u_3
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : InnerProductSpace 𝕜 E
        G : ι → Type u_4
        inst✝³ : (i : ι) → NormedAddCommGroup (G i)
        inst✝² : (i : ι) → InnerProductSpace 𝕜 (G i)
        inst✝¹ : CompleteSpace E
        V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
        inst✝ : ∀ (i : ι), CompleteSpace (G i)
        hVortho : OrthogonalFamily 𝕜 G V
        hVtotal : LE.le Top.top (iSup fun i => LinearMap.range (V i).toLinearMap).topo …
        ⊢ Function.Surjective ⇑hVortho.linearIsometry
      -/
      rw [← LinearIsometry.coe_toLinearMap]
      exact LinearMap.range_eq_top.mp
        (eq_top_iff.mpr <| hVtotal.trans_eq hVortho.range_linearIsometry.symm) }


/-- This is `Orthonormal.isHilbertSum` in the case of actual inclusions from subspaces. -/
theorem IsHilbertSum.mkInternal [∀ i, CompleteSpace <| F i]
    (hFortho : OrthogonalFamily 𝕜 (fun i => F i) fun i => (F i).subtypeₗᵢ)
    (hFtotal : ⊤ ≤ (⨆ i, F i).topologicalClosure) :
    IsHilbertSum 𝕜 (fun i => F i) fun i => (F i).subtypeₗᵢ :=
                              /-
                                ι : Type u_1
                                𝕜 : Type u_2
                                inst✝⁴ : RCLike 𝕜
                                E : Type u_3
                                inst✝³ : NormedAddCommGroup E
                                inst✝² : InnerProductSpace 𝕜 E
                                inst✝¹ : CompleteSpace E
                                F : ι → Submodule 𝕜 E
                                inst✝ : ∀ (i : ι), CompleteSpace (Subtype fun x => Membership.mem (F i) x)
                                hFortho : OrthogonalFamily 𝕜 (fun i => Subtype fun x => Membership.mem (F i) x …
                                hFtotal : LE.le Top.top (iSup fun i => F i).topologicalClosure
                                ⊢ LE.le Top.top (iSup fun i => LinearMap.range (F i).subtypeₗᵢ.toLinearMap).to …
                              -/
  IsHilbertSum.mk hFortho (by simpa [subtypeₗᵢ_toLinearMap, range_subtype] using hFtotal)
                              /-
                                🎉 no goals
                              -/


/-- *A* Hilbert sum `(E, V)` of `G` is canonically isomorphic to *the* Hilbert sum of `G`,
i.e `lp G 2`.

Note that this goes in the opposite direction from `OrthogonalFamily.linearIsometry`. -/
noncomputable def IsHilbertSum.linearIsometryEquiv (hV : IsHilbertSum 𝕜 G V) : E ≃ₗᵢ[𝕜] lp G 2 :=
  LinearIsometryEquiv.symm <|
    LinearIsometryEquiv.ofSurjective hV.OrthogonalFamily.linearIsometry hV.surjective_isometry


/-- In the canonical isometric isomorphism between a Hilbert sum `E` of `G` and `lp G 2`,
a vector `w : lp G 2` is the image of the infinite sum of the associated elements in `E`. -/
protected theorem IsHilbertSum.linearIsometryEquiv_symm_apply (hV : IsHilbertSum 𝕜 G V)
    (w : lp G 2) : hV.linearIsometryEquiv.symm w = ∑' i, V i (w i) := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝⁵ : RCLike 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    G : ι → Type u_4
    inst✝² : (i : ι) → NormedAddCommGroup (G i)
    inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝ : CompleteSpace E
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : IsHilbertSum 𝕜 G V
    w : Subtype fun x => Membership.mem (lp G 2) x
    ⊢ Eq (hV.linearIsometryEquiv.symm w) (tsum fun i => (V i) (↑w i))
  -/
  simp [IsHilbertSum.linearIsometryEquiv, OrthogonalFamily.linearIsometry_apply]
  /-
    🎉 no goals
  -/


/-- In the canonical isometric isomorphism between a Hilbert sum `E` of `G` and `lp G 2`,
a vector `w : lp G 2` is the image of the infinite sum of the associated elements in `E`, and this
sum indeed converges. -/
protected theorem IsHilbertSum.hasSum_linearIsometryEquiv_symm (hV : IsHilbertSum 𝕜 G V)
    (w : lp G 2) : HasSum (fun i => V i (w i)) (hV.linearIsometryEquiv.symm w) := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝⁵ : RCLike 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : InnerProductSpace 𝕜 E
    G : ι → Type u_4
    inst✝² : (i : ι) → NormedAddCommGroup (G i)
    inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝ : CompleteSpace E
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    hV : IsHilbertSum 𝕜 G V
    w : Subtype fun x => Membership.mem (lp G 2) x
    ⊢ HasSum (fun i => (V i) (↑w i)) (hV.linearIsometryEquiv.symm w)
  -/
  simp [IsHilbertSum.linearIsometryEquiv, OrthogonalFamily.hasSum_linearIsometry]
  /-
    🎉 no goals
  -/


/-- In the canonical isometric isomorphism between a Hilbert sum `E` of `G : ι → Type*` and
`lp G 2`, an "elementary basis vector" in `lp G 2` supported at `i : ι` is the image of the
associated element in `E`. -/
@[simp]
protected theorem IsHilbertSum.linearIsometryEquiv_symm_apply_single
    [DecidableEq ι] (hV : IsHilbertSum 𝕜 G V) {i : ι} (x : G i) :
    hV.linearIsometryEquiv.symm (lp.single 2 i x) = V i x := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝⁶ : RCLike 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : InnerProductSpace 𝕜 E
    G : ι → Type u_4
    inst✝³ : (i : ι) → NormedAddCommGroup (G i)
    inst✝² : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝¹ : CompleteSpace E
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    inst✝ : DecidableEq ι
    hV : IsHilbertSum 𝕜 G V
    i : ι
    x : G i
    ⊢ Eq (hV.linearIsometryEquiv.symm (lp.single 2 i x)) ((V i) x)
  -/
  simp [IsHilbertSum.linearIsometryEquiv, OrthogonalFamily.linearIsometry_apply_single]
  /-
    🎉 no goals
  -/


/-- In the canonical isometric isomorphism between a Hilbert sum `E` of `G : ι → Type*` and
`lp G 2`, a finitely-supported vector in `lp G 2` is the image of the associated finite sum of
elements of `E`. -/
protected theorem IsHilbertSum.linearIsometryEquiv_symm_apply_dfinsupp_sum_single
    [DecidableEq ι] [∀ i, DecidableEq (G i)] (hV : IsHilbertSum 𝕜 G V) (W₀ : Π₀ i : ι, G i) :
    hV.linearIsometryEquiv.symm (W₀.sum (lp.single 2)) = W₀.sum fun i => V i := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝⁷ : RCLike 𝕜
    E : Type u_3
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : InnerProductSpace 𝕜 E
    G : ι → Type u_4
    inst✝⁴ : (i : ι) → NormedAddCommGroup (G i)
    inst✝³ : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝² : CompleteSpace E
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    hV : IsHilbertSum 𝕜 G V
    W₀ : DFinsupp fun i => G i
    ⊢ Eq (hV.linearIsometryEquiv.symm (W₀.sum (lp.single 2))) (W₀.sum fun i => ⇑(V …
  -/
  simp only [map_dfinsupp_sum, IsHilbertSum.linearIsometryEquiv_symm_apply_single]
  /-
    🎉 no goals
  -/


/-- In the canonical isometric isomorphism between a Hilbert sum `E` of `G : ι → Type*` and
`lp G 2`, a finitely-supported vector in `lp G 2` is the image of the associated finite sum of
elements of `E`. -/
@[simp]
protected theorem IsHilbertSum.linearIsometryEquiv_apply_dfinsupp_sum_single
    [DecidableEq ι] [∀ i, DecidableEq (G i)] (hV : IsHilbertSum 𝕜 G V) (W₀ : Π₀ i : ι, G i) :
    ((W₀.sum (γ := lp G 2) fun a b ↦ hV.linearIsometryEquiv (V a b)) : ∀ i, G i) = W₀ := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝⁷ : RCLike 𝕜
    E : Type u_3
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : InnerProductSpace 𝕜 E
    G : ι → Type u_4
    inst✝⁴ : (i : ι) → NormedAddCommGroup (G i)
    inst✝³ : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝² : CompleteSpace E
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    hV : IsHilbertSum 𝕜 G V
    W₀ : DFinsupp fun i => G i
    ⊢ Eq ↑(W₀.sum fun a b => hV.linearIsometryEquiv ((V a) b)) ⇑W₀
  -/
  rw [← map_dfinsupp_sum]
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝⁷ : RCLike 𝕜
    E : Type u_3
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : InnerProductSpace 𝕜 E
    G : ι → Type u_4
    inst✝⁴ : (i : ι) → NormedAddCommGroup (G i)
    inst✝³ : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝² : CompleteSpace E
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    hV : IsHilbertSum 𝕜 G V
    W₀ : DFinsupp fun i => G i
    ⊢ Eq ↑(hV.linearIsometryEquiv (W₀.sum fun a => ⇑(V a))) ⇑W₀
  -/
  rw [← hV.linearIsometryEquiv_symm_apply_dfinsupp_sum_single]
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝⁷ : RCLike 𝕜
    E : Type u_3
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : InnerProductSpace 𝕜 E
    G : ι → Type u_4
    inst✝⁴ : (i : ι) → NormedAddCommGroup (G i)
    inst✝³ : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝² : CompleteSpace E
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    hV : IsHilbertSum 𝕜 G V
    W₀ : DFinsupp fun i => G i
    ⊢ Eq ↑(hV.linearIsometryEquiv (hV.linearIsometryEquiv.symm (W₀.sum (lp.single  …
  -/
  rw [LinearIsometryEquiv.apply_symm_apply]
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝⁷ : RCLike 𝕜
    E : Type u_3
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : InnerProductSpace 𝕜 E
    G : ι → Type u_4
    inst✝⁴ : (i : ι) → NormedAddCommGroup (G i)
    inst✝³ : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝² : CompleteSpace E
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    hV : IsHilbertSum 𝕜 G V
    W₀ : DFinsupp fun i => G i
    ⊢ Eq ↑(W₀.sum (lp.single 2)) ⇑W₀
  -/
  ext i
  /-
    case h
    ι : Type u_1
    𝕜 : Type u_2
    inst✝⁷ : RCLike 𝕜
    E : Type u_3
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : InnerProductSpace 𝕜 E
    G : ι → Type u_4
    inst✝⁴ : (i : ι) → NormedAddCommGroup (G i)
    inst✝³ : (i : ι) → InnerProductSpace 𝕜 (G i)
    inst✝² : CompleteSpace E
    V : (i : ι) → LinearIsometry (RingHom.id 𝕜) (G i) E
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (G i)
    hV : IsHilbertSum 𝕜 G V
    W₀ : DFinsupp fun i => G i
    i : ι
    ⊢ Eq (↑(W₀.sum (lp.single 2)) i) (W₀ i)
  -/
  simp +contextual [DFinsupp.sum, lp.single_apply]
  /-
    🎉 no goals
  -/


/-- Given a total orthonormal family `v : ι → E`, `E` is a Hilbert sum of `fun i : ι => 𝕜`
relative to the family of linear isometries `fun i k => k • v i`. -/
theorem Orthonormal.isHilbertSum {v : ι → E} (hv : Orthonormal 𝕜 v)
    (hsp : ⊤ ≤ (span 𝕜 (Set.range v)).topologicalClosure) :
    IsHilbertSum 𝕜 (fun _ : ι => 𝕜) fun i => LinearIsometry.toSpanSingleton 𝕜 E (hv.1 i) :=
  IsHilbertSum.mk hv.orthogonalFamily (by
    /-
      ι : Type u_1
      𝕜 : Type u_2
      inst✝³ : RCLike 𝕜
      E : Type u_3
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      inst✝ : CompleteSpace E
      v : ι → E
      hv : Orthonormal 𝕜 v
      hsp : LE.le Top.top (Submodule.span 𝕜 (Set.range v)).topologicalClosure
      ⊢ LE.le Top.top (iSup fun i => LinearMap.range (LinearIsometry.toSpanSingleton …
    -/
    convert hsp
    /-
      case h.e'_4.h.e'_9
      ι : Type u_1
      𝕜 : Type u_2
      inst✝³ : RCLike 𝕜
      E : Type u_3
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      inst✝ : CompleteSpace E
      v : ι → E
      hv : Orthonormal 𝕜 v
      hsp : LE.le Top.top (Submodule.span 𝕜 (Set.range v)).topologicalClosure
      ⊢ Eq (iSup fun i => LinearMap.range (LinearIsometry.toSpanSingleton 𝕜 E ⋯).toL …
    -/
    simp [← LinearMap.span_singleton_eq_range, ← Submodule.span_iUnion])
    /-
      🎉 no goals
    -/


theorem Submodule.isHilbertSumOrthogonal (K : Submodule 𝕜 E) [hK : CompleteSpace K] :
    IsHilbertSum 𝕜 (fun b => ↥(cond b K Kᗮ)) fun b => (cond b K Kᗮ).subtypeₗᵢ := by
  have : ∀ b, CompleteSpace (↥(cond b K Kᗮ)) := by
    intro b
    cases b <;> first | exact instOrthogonalCompleteSpace K | assumption
  /-
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    K : Submodule 𝕜 E
    hK : CompleteSpace (Subtype fun x => Membership.mem K x)
    this : ∀ (b : Bool), CompleteSpace (Subtype fun x => Membership.mem (cond b K  …
    ⊢ IsHilbertSum 𝕜 (fun b => Subtype fun x => Membership.mem (cond b K K.orthogo …
  -/
  refine IsHilbertSum.mkInternal _ K.orthogonalFamily_self ?_
  /-
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    K : Submodule 𝕜 E
    hK : CompleteSpace (Subtype fun x => Membership.mem K x)
    this : ∀ (b : Bool), CompleteSpace (Subtype fun x => Membership.mem (cond b K  …
    ⊢ LE.le Top.top (iSup fun i => cond i K K.orthogonal).topologicalClosure
  -/
  refine le_trans ?_ (Submodule.le_topologicalClosure _)
  /-
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    K : Submodule 𝕜 E
    hK : CompleteSpace (Subtype fun x => Membership.mem K x)
    this : ∀ (b : Bool), CompleteSpace (Subtype fun x => Membership.mem (cond b K  …
    ⊢ LE.le Top.top (iSup fun i => cond i K K.orthogonal)
  -/
  rw [iSup_bool_eq, cond, cond]
  /-
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    K : Submodule 𝕜 E
    hK : CompleteSpace (Subtype fun x => Membership.mem K x)
    this : ∀ (b : Bool), CompleteSpace (Subtype fun x => Membership.mem (cond b K  …
    ⊢ LE.le Top.top (Max.max K K.orthogonal)
  -/
  refine Codisjoint.top_le ?_
  /-
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    K : Submodule 𝕜 E
    hK : CompleteSpace (Subtype fun x => Membership.mem K x)
    this : ∀ (b : Bool), CompleteSpace (Subtype fun x => Membership.mem (cond b K  …
    ⊢ Codisjoint K K.orthogonal
  -/
  exact Submodule.isCompl_orthogonal_of_completeSpace.codisjoint
  /-
    🎉 no goals
  -/


/-- A Hilbert basis on `ι` for an inner product space `E` is an identification of `E` with the `lp`
space `ℓ²(ι, 𝕜)`. -/
structure HilbertBasis where ofRepr ::
  /-- The linear isometric equivalence implementing identifying the Hilbert space with `ℓ²`. -/
  repr : E ≃ₗᵢ[𝕜] ℓ²(ι, 𝕜)


instance {ι : Type*} : Inhabited (HilbertBasis ι 𝕜 ℓ²(ι, 𝕜)) :=
  ⟨ofRepr (LinearIsometryEquiv.refl 𝕜 _)⟩


open Classical in
/-- `b i` is the `i`th basis vector. -/
instance instCoeFun : CoeFun (HilbertBasis ι 𝕜 E) fun _ => ι → E where
  coe b i := b.repr.symm (lp.single 2 i (1 : 𝕜))

-- This is a bad `@[simp]` lemma: the RHS is a coercion containing the LHS.

protected theorem repr_symm_single [DecidableEq ι] (b : HilbertBasis ι 𝕜 E) (i : ι) :
    b.repr.symm (lp.single 2 i (1 : 𝕜)) = b i := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : DecidableEq ι
    b : HilbertBasis ι 𝕜 E
    i : ι
    ⊢ Eq (b.repr.symm (lp.single 2 i 1)) ((fun i => b.repr.symm (lp.single 2 i 1)) …
  -/
  convert rfl
  /-
    🎉 no goals
  -/


protected theorem repr_self [DecidableEq ι] (b : HilbertBasis ι 𝕜 E) (i : ι) :
    b.repr (b i) = lp.single 2 i (1 : 𝕜) := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : DecidableEq ι
    b : HilbertBasis ι 𝕜 E
    i : ι
    ⊢ Eq (b.repr ((fun i => b.repr.symm (lp.single 2 i 1)) i)) (lp.single 2 i 1)
  -/
  simp only [LinearIsometryEquiv.apply_symm_apply]
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : DecidableEq ι
    b : HilbertBasis ι 𝕜 E
    i : ι
    ⊢ Eq (lp.single 2 i 1) (lp.single 2 i 1)
  -/
  convert rfl
  /-
    🎉 no goals
  -/


protected theorem repr_apply_apply (b : HilbertBasis ι 𝕜 E) (v : E) (i : ι) :
    b.repr v i = ⟪b i, v⟫ := by
  classical
  rw [← b.repr.inner_map_map (b i) v, b.repr_self, lp.inner_single_left]
  simp


@[simp]
protected theorem orthonormal (b : HilbertBasis ι 𝕜 E) : Orthonormal 𝕜 b := by
  classical
  rw [orthonormal_iff_ite]
  intro i j
  rw [← b.repr.inner_map_map (b i) (b j), b.repr_self, b.repr_self, lp.inner_single_left,
    lp.single_apply]
  simp


protected theorem hasSum_repr_symm (b : HilbertBasis ι 𝕜 E) (f : ℓ²(ι, 𝕜)) :
    HasSum (fun i => f i • b i) (b.repr.symm f) := by
  classical
  suffices H : (fun i : ι => f i • b i) = fun b_1 : ι => b.repr.symm.toContinuousLinearEquiv <|
      (fun i : ι => lp.single 2 i (f i) (E := (fun _ : ι => 𝕜))) b_1 by
    rw [H]
    have : HasSum (fun i : ι => lp.single 2 i (f i)) f := lp.hasSum_single ENNReal.two_ne_top f
    exact (↑b.repr.symm.toContinuousLinearEquiv : ℓ²(ι, 𝕜) →L[𝕜] E).hasSum this
  ext i
  apply b.repr.injective
  letI : NormedSpace 𝕜 (lp (fun _i : ι => 𝕜) 2) := by infer_instance
  have : lp.single (E := (fun _ : ι => 𝕜)) 2 i (f i * 1) = f i • lp.single 2 i 1 :=
    lp.single_smul (E := (fun _ : ι => 𝕜)) 2 i (1 : 𝕜) (f i)
  rw [mul_one] at this
  rw [LinearIsometryEquiv.map_smul, b.repr_self, ← this,
    LinearIsometryEquiv.coe_toContinuousLinearEquiv]
  exact (b.repr.apply_symm_apply (lp.single 2 i (f i))).symm


protected theorem hasSum_repr (b : HilbertBasis ι 𝕜 E) (x : E) :
                                               /-
                                                 ι : Type u_1
                                                 𝕜 : Type u_2
                                                 inst✝² : RCLike 𝕜
                                                 E : Type u_3
                                                 inst✝¹ : NormedAddCommGroup E
                                                 inst✝ : InnerProductSpace 𝕜 E
                                                 b : HilbertBasis ι 𝕜 E
                                                 x : E
                                                 ⊢ HasSum (fun i => HSMul.hSMul (↑(b.repr x) i) ((fun i => b.repr.symm (lp.sing …
                                               -/
    HasSum (fun i => b.repr x i • b i) x := by simpa using b.hasSum_repr_symm (b.repr x)
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
protected theorem dense_span (b : HilbertBasis ι 𝕜 E) :
    (span 𝕜 (Set.range b)).topologicalClosure = ⊤ := by
  classical
    rw [eq_top_iff]
    rintro x -
    refine mem_closure_of_tendsto (b.hasSum_repr x) (Eventually.of_forall ?_)
    intro s
    simp only [SetLike.mem_coe]
    refine sum_mem ?_
    rintro i -
    refine smul_mem _ _ ?_
    exact subset_span ⟨i, rfl⟩


protected theorem hasSum_inner_mul_inner (b : HilbertBasis ι 𝕜 E) (x y : E) :
    HasSum (fun i => ⟪x, b i⟫ * ⟪b i, y⟫) ⟪x, y⟫ := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    b : HilbertBasis ι 𝕜 E
    x y : E
    ⊢ HasSum (fun i => HMul.hMul (Inner.inner x ((fun i => b.repr.symm (lp.single  …
  -/
  convert (b.hasSum_repr y).mapL (innerSL 𝕜 x) using 1
  /-
    case h.e'_5
    ι : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    b : HilbertBasis ι 𝕜 E
    x y : E
    ⊢ Eq (fun i => HMul.hMul (Inner.inner x ((fun i => b.repr.symm (lp.single 2 i  …
  -/
  ext i
  /-
    case h.e'_5.h
    ι : Type u_1
    𝕜 : Type u_2
    inst✝² : RCLike 𝕜
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    b : HilbertBasis ι 𝕜 E
    x y : E
    i : ι
    ⊢ Eq (HMul.hMul (Inner.inner x ((fun i => b.repr.symm (lp.single 2 i 1)) i)) ( …
  -/
  rw [innerSL_apply, b.repr_apply_apply, inner_smul_right, mul_comm]
  /-
    🎉 no goals
  -/


protected theorem summable_inner_mul_inner (b : HilbertBasis ι 𝕜 E) (x y : E) :
    Summable fun i => ⟪x, b i⟫ * ⟪b i, y⟫ :=
  (b.hasSum_inner_mul_inner x y).summable


protected theorem tsum_inner_mul_inner (b : HilbertBasis ι 𝕜 E) (x y : E) :
    ∑' i, ⟪x, b i⟫ * ⟪b i, y⟫ = ⟪x, y⟫ :=
  (b.hasSum_inner_mul_inner x y).tsum_eq

-- Note: this should be `b.repr` composed with an identification of `lp (fun i : ι => 𝕜) p` with
-- `PiLp p (fun i : ι => 𝕜)` (in this case with `p = 2`), but we don't have this yet (July 2022).

/-- A finite Hilbert basis is an orthonormal basis. -/
protected def toOrthonormalBasis [Fintype ι] (b : HilbertBasis ι 𝕜 E) : OrthonormalBasis ι 𝕜 E :=
  OrthonormalBasis.mk b.orthonormal
    (by
      /-
        ι : Type u_1
        𝕜 : Type u_2
        inst✝⁵ : RCLike 𝕜
        E : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : InnerProductSpace 𝕜 E
        G : ι → Type u_4
        inst✝² : (i : ι) → NormedAddCommGroup (G i)
        inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
        inst✝ : Fintype ι
        b : HilbertBasis ι 𝕜 E
        ⊢ LE.le Top.top (Submodule.span 𝕜 (Set.range fun i => b.repr.symm (lp.single 2 …
      -/
      refine Eq.ge ?_
      classical
      have := (span 𝕜 (Finset.univ.image b : Set E)).closed_of_finiteDimensional
      simpa only [Finset.coe_image, Finset.coe_univ, Set.image_univ, HilbertBasis.dense_span] using
        this.submodule_topologicalClosure_eq.symm)


@[simp]
theorem coe_toOrthonormalBasis [Fintype ι] (b : HilbertBasis ι 𝕜 E) :
    (b.toOrthonormalBasis : ι → E) = b :=
  OrthonormalBasis.coe_mk _ _


protected theorem hasSum_orthogonalProjection {U : Submodule 𝕜 E} [CompleteSpace U]
    (b : HilbertBasis ι 𝕜 U) (x : E) :
    HasSum (fun i => ⟪(b i : E), x⟫ • b i) (orthogonalProjection U x) := by
  simpa only [b.repr_apply_apply, inner_orthogonalProjection_eq_of_mem_left] using
    b.hasSum_repr (orthogonalProjection U x)


theorem finite_spans_dense [DecidableEq E] (b : HilbertBasis ι 𝕜 E) :
    (⨆ J : Finset ι, span 𝕜 (J.image b : Set E)).topologicalClosure = ⊤ :=
  eq_top_iff.mpr <| b.dense_span.ge.trans (by
    /-
      ι : Type u_1
      𝕜 : Type u_2
      inst✝³ : RCLike 𝕜
      E : Type u_3
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      inst✝ : DecidableEq E
      b : HilbertBasis ι 𝕜 E
      ⊢ LE.le (Submodule.span 𝕜 (Set.range fun i => b.repr.symm (lp.single 2 i 1))). …
    -/
    simp_rw [← Submodule.span_iUnion]
    exact topologicalClosure_mono (span_mono <| Set.range_subset_iff.mpr fun i =>
      Set.mem_iUnion_of_mem {i} <| Finset.mem_coe.mpr <| Finset.mem_image_of_mem _ <|
      Finset.mem_singleton_self i))


/-- An orthonormal family of vectors whose span is dense in the whole module is a Hilbert basis. -/
protected def mk (hsp : ⊤ ≤ (span 𝕜 (Set.range v)).topologicalClosure) : HilbertBasis ι 𝕜 E :=
  HilbertBasis.ofRepr <| (hv.isHilbertSum hsp).linearIsometryEquiv


theorem _root_.Orthonormal.linearIsometryEquiv_symm_apply_single_one [DecidableEq ι] (h i) :
    (hv.isHilbertSum h).linearIsometryEquiv.symm (lp.single 2 i 1) = v i := by
  rw [IsHilbertSum.linearIsometryEquiv_symm_apply_single, LinearIsometry.toSpanSingleton_apply,
    one_smul]


@[simp]
protected theorem coe_mk (hsp : ⊤ ≤ (span 𝕜 (Set.range v)).topologicalClosure) :
    ⇑(HilbertBasis.mk hv hsp) = v := by
  classical
  apply funext <| Orthonormal.linearIsometryEquiv_symm_apply_single_one hv hsp


/-- An orthonormal family of vectors whose span has trivial orthogonal complement is a Hilbert
basis. -/
protected def mkOfOrthogonalEqBot (hsp : (span 𝕜 (Set.range v))ᗮ = ⊥) : HilbertBasis ι 𝕜 E :=
  HilbertBasis.mk hv
        /-
          ι : Type u_1
          𝕜 : Type u_2
          inst✝⁵ : RCLike 𝕜
          E : Type u_3
          inst✝⁴ : NormedAddCommGroup E
          inst✝³ : InnerProductSpace 𝕜 E
          G : ι → Type u_4
          inst✝² : (i : ι) → NormedAddCommGroup (G i)
          inst✝¹ : (i : ι) → InnerProductSpace 𝕜 (G i)
          inst✝ : CompleteSpace E
          v : ι → E
          hv : Orthonormal 𝕜 v
          hsp : Eq (Submodule.span 𝕜 (Set.range v)).orthogonal Bot.bot
          ⊢ LE.le Top.top (Submodule.span 𝕜 (Set.range v)).topologicalClosure
        -/
    (by rw [← orthogonal_orthogonal_eq_closure, ← eq_top_iff, orthogonal_eq_top_iff, hsp])
        /-
          🎉 no goals
        -/


@[simp]
protected theorem coe_mkOfOrthogonalEqBot (hsp : (span 𝕜 (Set.range v))ᗮ = ⊥) :
    ⇑(HilbertBasis.mkOfOrthogonalEqBot hv hsp) = v :=
  HilbertBasis.coe_mk hv _

-- Note : this should be `b.repr` composed with an identification of `lp (fun i : ι => 𝕜) p` with
-- `PiLp p (fun i : ι => 𝕜)` (in this case with `p = 2`), but we don't have this yet (July 2022).

/-- An orthonormal basis is a Hilbert basis. -/
protected def _root_.OrthonormalBasis.toHilbertBasis [Fintype ι] (b : OrthonormalBasis ι 𝕜 E) :
    HilbertBasis ι 𝕜 E :=
  HilbertBasis.mk b.orthonormal <| by
    simpa only [← OrthonormalBasis.coe_toBasis, b.toBasis.span_eq, eq_top_iff] using
      @subset_closure E _ _


@[simp]
theorem _root_.OrthonormalBasis.coe_toHilbertBasis [Fintype ι] (b : OrthonormalBasis ι 𝕜 E) :
    (b.toHilbertBasis : ι → E) = b :=
  HilbertBasis.coe_mk _ _


/-- A Hilbert space admits a Hilbert basis extending a given orthonormal subset. -/
theorem _root_.Orthonormal.exists_hilbertBasis_extension {s : Set E}
    (hs : Orthonormal 𝕜 ((↑) : s → E)) :
    ∃ (w : Set E) (b : HilbertBasis w 𝕜 E), s ⊆ w ∧ ⇑b = ((↑) : w → E) :=
  let ⟨w, hws, hw_ortho, hw_max⟩ := exists_maximal_orthonormal hs
  ⟨w, HilbertBasis.mkOfOrthogonalEqBot hw_ortho
    (by simpa only [Subtype.range_coe_subtype, Set.setOf_mem_eq,
      maximal_orthonormal_iff_orthogonalComplement_eq_bot hw_ortho] using hw_max),
    hws, HilbertBasis.coe_mkOfOrthogonalEqBot _ _⟩


/-- A Hilbert space admits a Hilbert basis. -/
theorem _root_.exists_hilbertBasis : ∃ (w : Set E) (b : HilbertBasis w 𝕜 E), ⇑b = ((↑) : w → E) :=
  let ⟨w, hw, _, hw''⟩ := (orthonormal_empty 𝕜 E).exists_hilbertBasis_extension
  ⟨w, hw, hw''⟩



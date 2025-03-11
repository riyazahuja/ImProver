/-- A term of `FormalMultilinearSeries.changeOriginSeries`.

Given a formal multilinear series `p` and a point `x` in its ball of convergence,
`p.changeOrigin x` is a formal multilinear series such that
`p.sum (x+y) = (p.changeOrigin x).sum y` when this makes sense. Each term of `p.changeOrigin x`
is itself an analytic function of `x` given by the series `p.changeOriginSeries`. Each term in
`changeOriginSeries` is the sum of `changeOriginSeriesTerm`'s over all `s` of cardinality `l`.
The definition is such that `p.changeOriginSeriesTerm k l s hs (fun _ ↦ x) (fun _ ↦ y) =
p (k + l) (s.piecewise (fun _ ↦ x) (fun _ ↦ y))`
-/
def changeOriginSeriesTerm (k l : ℕ) (s : Finset (Fin (k + l))) (hs : s.card = l) :
    E[×l]→L[𝕜] E[×k]→L[𝕜] F :=
  let a := ContinuousMultilinearMap.curryFinFinset 𝕜 E F hs
        /-
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          inst✝⁴ : NontriviallyNormedField 𝕜
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          p : FormalMultilinearSeries 𝕜 E F
          x y : E
          r : NNReal
          k l : Nat
          s : Finset (Fin (HAdd.hAdd k l))
          hs : Eq s.card l
          ⊢ Eq (HasCompl.compl s).card k
        -/
    (by rw [Finset.card_compl, Fintype.card_fin, hs, add_tsub_cancel_right])
        /-
          🎉 no goals
        -/
  a (p (k + l))


theorem changeOriginSeriesTerm_apply (k l : ℕ) (s : Finset (Fin (k + l))) (hs : s.card = l)
    (x y : E) :
    (p.changeOriginSeriesTerm k l s hs (fun _ => x) fun _ => y) =
      p (k + l) (s.piecewise (fun _ => x) fun _ => y) :=
  ContinuousMultilinearMap.curryFinFinset_apply_const _ _ _ _ _


@[simp]
theorem norm_changeOriginSeriesTerm (k l : ℕ) (s : Finset (Fin (k + l))) (hs : s.card = l) :
    ‖p.changeOriginSeriesTerm k l s hs‖ = ‖p (k + l)‖ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    k l : Nat
    s : Finset (Fin (HAdd.hAdd k l))
    hs : Eq s.card l
    ⊢ Eq (Norm.norm (p.changeOriginSeriesTerm k l s hs)) (Norm.norm (p (HAdd.hAdd  …
  -/
  simp only [changeOriginSeriesTerm, LinearIsometryEquiv.norm_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem nnnorm_changeOriginSeriesTerm (k l : ℕ) (s : Finset (Fin (k + l))) (hs : s.card = l) :
    ‖p.changeOriginSeriesTerm k l s hs‖₊ = ‖p (k + l)‖₊ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    k l : Nat
    s : Finset (Fin (HAdd.hAdd k l))
    hs : Eq s.card l
    ⊢ Eq (NNNorm.nnnorm (p.changeOriginSeriesTerm k l s hs)) (NNNorm.nnnorm (p (HA …
  -/
  simp only [changeOriginSeriesTerm, LinearIsometryEquiv.nnnorm_map]
  /-
    🎉 no goals
  -/


theorem nnnorm_changeOriginSeriesTerm_apply_le (k l : ℕ) (s : Finset (Fin (k + l)))
    (hs : s.card = l) (x y : E) :
    ‖p.changeOriginSeriesTerm k l s hs (fun _ => x) fun _ => y‖₊ ≤
      ‖p (k + l)‖₊ * ‖x‖₊ ^ l * ‖y‖₊ ^ k := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    k l : Nat
    s : Finset (Fin (HAdd.hAdd k l))
    hs : Eq s.card l
    x y : E
    ⊢ LE.le (NNNorm.nnnorm (((p.changeOriginSeriesTerm k l s hs) fun x_1 => x) fun …
  -/
  rw [← p.nnnorm_changeOriginSeriesTerm k l s hs, ← Fin.prod_const, ← Fin.prod_const]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    k l : Nat
    s : Finset (Fin (HAdd.hAdd k l))
    hs : Eq s.card l
    x y : E
    ⊢ LE.le (NNNorm.nnnorm (((p.changeOriginSeriesTerm k l s hs) fun x_1 => x) fun …
  -/
  apply ContinuousMultilinearMap.le_of_opNNNorm_le
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    k l : Nat
    s : Finset (Fin (HAdd.hAdd k l))
    hs : Eq s.card l
    x y : E
    ⊢ LE.le (NNNorm.nnnorm ((p.changeOriginSeriesTerm k l s hs) fun x_1 => x)) (HM …
  -/
  apply ContinuousMultilinearMap.le_opNNNorm
  /-
    🎉 no goals
  -/


/-- The power series for `f.changeOrigin k`.

Given a formal multilinear series `p` and a point `x` in its ball of convergence,
`p.changeOrigin x` is a formal multilinear series such that
`p.sum (x+y) = (p.changeOrigin x).sum y` when this makes sense. Its `k`-th term is the sum of
the series `p.changeOriginSeries k`. -/
def changeOriginSeries (k : ℕ) : FormalMultilinearSeries 𝕜 E (E[×k]→L[𝕜] F) := fun l =>
  ∑ s : { s : Finset (Fin (k + l)) // Finset.card s = l }, p.changeOriginSeriesTerm k l s s.2


theorem nnnorm_changeOriginSeries_le_tsum (k l : ℕ) :
    ‖p.changeOriginSeries k l‖₊ ≤
      ∑' _ : { s : Finset (Fin (k + l)) // s.card = l }, ‖p (k + l)‖₊ :=
  (nnnorm_sum_le _ (fun t => changeOriginSeriesTerm p k l (Subtype.val t) t.prop)).trans_eq <| by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      k l : Nat
      ⊢ Eq (Finset.univ.sum fun a => NNNorm.nnnorm (p.changeOriginSeriesTerm k l ↑a  …
    -/
    simp_rw [tsum_fintype, nnnorm_changeOriginSeriesTerm (p := p) (k := k) (l := l)]
    /-
      🎉 no goals
    -/


theorem nnnorm_changeOriginSeries_apply_le_tsum (k l : ℕ) (x : E) :
    ‖p.changeOriginSeries k l fun _ => x‖₊ ≤
      ∑' _ : { s : Finset (Fin (k + l)) // s.card = l }, ‖p (k + l)‖₊ * ‖x‖₊ ^ l := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    k l : Nat
    x : E
    ⊢ LE.le (NNNorm.nnnorm ((p.changeOriginSeries k l) fun x_1 => x)) (tsum fun x_ …
  -/
  rw [NNReal.tsum_mul_right, ← Fin.prod_const]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    k l : Nat
    x : E
    ⊢ LE.le (NNNorm.nnnorm ((p.changeOriginSeries k l) fun x_1 => x)) (HMul.hMul ( …
  -/
  exact (p.changeOriginSeries k l).le_of_opNNNorm_le (p.nnnorm_changeOriginSeries_le_tsum _ _) _
  /-
    🎉 no goals
  -/


/-- Changing the origin of a formal multilinear series `p`, so that
`p.sum (x+y) = (p.changeOrigin x).sum y` when this makes sense.
-/
def changeOrigin (x : E) : FormalMultilinearSeries 𝕜 E F :=
  fun k => (p.changeOriginSeries k).sum x


/-- An auxiliary equivalence useful in the proofs about
`FormalMultilinearSeries.changeOriginSeries`: the set of triples `(k, l, s)`, where `s` is a
`Finset (Fin (k + l))` of cardinality `l` is equivalent to the set of pairs `(n, s)`, where `s` is a
`Finset (Fin n)`.

The forward map sends `(k, l, s)` to `(k + l, s)` and the inverse map sends `(n, s)` to
`(n - Finset.card s, Finset.card s, s)`. The actual definition is less readable because of problems
with non-definitional equalities. -/
@[simps]
def changeOriginIndexEquiv :
    (Σ k l : ℕ, { s : Finset (Fin (k + l)) // s.card = l }) ≃ Σ n : ℕ, Finset (Fin n) where
  toFun s := ⟨s.1 + s.2.1, s.2.2⟩
  invFun s :=
    ⟨s.1 - s.2.card, s.2.card,
      ⟨s.2.map
        (finCongr <| (tsub_add_cancel_of_le <| card_finset_fin_le s.2).symm).toEmbedding,
        Finset.card_map _⟩⟩
  left_inv := by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x y : E
      r : NNReal
      ⊢ Function.LeftInverse (fun s => ⟨HSub.hSub s.fst s.snd.card, ⟨s.snd.card, ⟨Fi …
    -/
    rintro ⟨k, l, ⟨s : Finset (Fin <| k + l), hs : s.card = l⟩⟩
    /-
      case mk.mk.mk
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x y : E
      r : NNReal
      k l : Nat
      s : Finset (Fin (HAdd.hAdd k l))
      hs : Eq s.card l
      ⊢ Eq ((fun s => ⟨HSub.hSub s.fst s.snd.card, ⟨s.snd.card, ⟨Finset.map (finCong …
    -/
    dsimp only [Subtype.coe_mk]
    -- Lean can't automatically generalize `k' = k + l - s.card`, `l' = s.card`, so we explicitly
    -- formulate the generalized goal
    suffices ∀ k' l', k' = k → l' = l → ∀ (hkl : k + l = k' + l') (hs'),
        (⟨k', l', ⟨s.map (finCongr hkl).toEmbedding, hs'⟩⟩ :
          Σk l : ℕ, { s : Finset (Fin (k + l)) // s.card = l }) = ⟨k, l, ⟨s, hs⟩⟩ by
      apply this <;> simp only [hs, add_tsub_cancel_right]
    /-
      case mk.mk.mk
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x y : E
      r : NNReal
      k l : Nat
      s : Finset (Fin (HAdd.hAdd k l))
      hs : Eq s.card l
      ⊢ ∀ (k' l' : Nat), Eq k' k → Eq l' l → ∀ (hkl : Eq (HAdd.hAdd k l) (HAdd.hAdd  …
    -/
    rintro _ _ rfl rfl hkl hs'
    simp only [Equiv.refl_toEmbedding, finCongr_refl, Finset.map_refl, eq_self_iff_true,
      OrderIso.refl_toEquiv, and_self_iff, heq_iff_eq]
  right_inv := by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x y : E
      r : NNReal
      ⊢ Function.RightInverse (fun s => ⟨HSub.hSub s.fst s.snd.card, ⟨s.snd.card, ⟨F …
    -/
    rintro ⟨n, s⟩
    /-
      case mk
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      x y : E
      r : NNReal
      n : Nat
      s : Finset (Fin n)
      ⊢ Eq ((fun s => ⟨HAdd.hAdd s.fst s.snd.fst, ↑s.snd.snd⟩) ((fun s => ⟨HSub.hSub …
    -/
    simp [tsub_add_cancel_of_le (card_finset_fin_le s), finCongr_eq_equivCast]
    /-
      🎉 no goals
    -/


lemma changeOriginSeriesTerm_changeOriginIndexEquiv_symm (n t) :
    let s := changeOriginIndexEquiv.symm ⟨n, t⟩
    p.changeOriginSeriesTerm s.1 s.2.1 s.2.2 s.2.2.2 (fun _ ↦ x) (fun _ ↦ y) =
    p n (t.piecewise (fun _ ↦ x) fun _ ↦ y) := by
  have : ∀ (m) (hm : n = m), p n (t.piecewise (fun _ ↦ x) fun _ ↦ y) =
      p m ((t.map (finCongr hm).toEmbedding).piecewise (fun _ ↦ x) fun _ ↦ y) := by
    rintro m rfl
    simp (config := { unfoldPartialApp := true }) [Finset.piecewise]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    x y : E
    n : Nat
    t : Finset (Fin n)
    this : ∀ (m : Nat) (hm : Eq n m), Eq ((p n) (t.piecewise (fun x_1 => x) fun x  …
    ⊢ let s := FormalMultilinearSeries.changeOriginIndexEquiv.symm ⟨n, t⟩;
      Eq (((p.changeOriginSeriesTerm s.fst s.snd.fst ↑s.snd.snd ⋯) fun x_1 => x) f …
  -/
  simp_rw [changeOriginSeriesTerm_apply, eq_comm]; apply this
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem changeOriginSeries_summable_aux₁ {r r' : ℝ≥0} (hr : (r + r' : ℝ≥0∞) < p.radius) :
    Summable fun s : Σk l : ℕ, { s : Finset (Fin (k + l)) // s.card = l } =>
      ‖p (s.1 + s.2.1)‖₊ * r ^ s.2.1 * r' ^ s.1 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r r' : NNReal
    hr : LT.lt (HAdd.hAdd ↑r ↑r') p.radius
    ⊢ Summable fun s => HMul.hMul (HMul.hMul (NNNorm.nnnorm (p (HAdd.hAdd s.fst s. …
  -/
  rw [← changeOriginIndexEquiv.symm.summable_iff]
  dsimp only [Function.comp_def, changeOriginIndexEquiv_symm_apply_fst,
    changeOriginIndexEquiv_symm_apply_snd_fst]
  have : ∀ n : ℕ,
      HasSum (fun s : Finset (Fin n) => ‖p (n - s.card + s.card)‖₊ * r ^ s.card * r' ^ (n - s.card))
        (‖p n‖₊ * (r + r') ^ n) := by
    intro n
    -- TODO: why `simp only [tsub_add_cancel_of_le (card_finset_fin_le _)]` fails?
    convert_to HasSum (fun s : Finset (Fin n) => ‖p n‖₊ * (r ^ s.card * r' ^ (n - s.card))) _
    · ext1 s
      rw [tsub_add_cancel_of_le (card_finset_fin_le _), mul_assoc]
    rw [← Fin.sum_pow_mul_eq_add_pow]
    exact (hasSum_fintype _).mul_left _
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r r' : NNReal
    hr : LT.lt (HAdd.hAdd ↑r ↑r') p.radius
    this : ∀ (n : Nat), HasSum (fun s => HMul.hMul (HMul.hMul (NNNorm.nnnorm (p (H …
    ⊢ Summable fun x => HMul.hMul (HMul.hMul (NNNorm.nnnorm (p (HAdd.hAdd (HSub.hS …
  -/
  refine NNReal.summable_sigma.2 ⟨fun n => (this n).summable, ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r r' : NNReal
    hr : LT.lt (HAdd.hAdd ↑r ↑r') p.radius
    this : ∀ (n : Nat), HasSum (fun s => HMul.hMul (HMul.hMul (NNNorm.nnnorm (p (H …
    ⊢ Summable fun x => tsum fun y => HMul.hMul (HMul.hMul (NNNorm.nnnorm (p (HAdd …
  -/
  simp only [(this _).tsum_eq]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r r' : NNReal
    hr : LT.lt (HAdd.hAdd ↑r ↑r') p.radius
    this : ∀ (n : Nat), HasSum (fun s => HMul.hMul (HMul.hMul (NNNorm.nnnorm (p (H …
    ⊢ Summable fun x => HMul.hMul (NNNorm.nnnorm (p x)) (HPow.hPow (HAdd.hAdd r r' …
  -/
  exact p.summable_nnnorm_mul_pow hr
  /-
    🎉 no goals
  -/


theorem changeOriginSeries_summable_aux₂ (hr : (r : ℝ≥0∞) < p.radius) (k : ℕ) :
    Summable fun s : Σl : ℕ, { s : Finset (Fin (k + l)) // s.card = l } =>
      ‖p (k + s.1)‖₊ * r ^ s.1 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : NNReal
    hr : LT.lt (↑r) p.radius
    k : Nat
    ⊢ Summable fun s => HMul.hMul (NNNorm.nnnorm (p (HAdd.hAdd k s.fst))) (HPow.hP …
  -/
  rcases ENNReal.lt_iff_exists_add_pos_lt.1 hr with ⟨r', h0, hr'⟩
  simpa only [mul_inv_cancel_right₀ (pow_pos h0 _).ne'] using
    ((NNReal.summable_sigma.1 (p.changeOriginSeries_summable_aux₁ hr')).1 k).mul_right (r' ^ k)⁻¹


theorem changeOriginSeries_summable_aux₃ {r : ℝ≥0} (hr : ↑r < p.radius) (k : ℕ) :
    Summable fun l : ℕ => ‖p.changeOriginSeries k l‖₊ * r ^ l := by
  refine NNReal.summable_of_le
    (fun n => ?_) (NNReal.summable_sigma.1 <| p.changeOriginSeries_summable_aux₂ hr k).2
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : NNReal
    hr : LT.lt (↑r) p.radius
    k n : Nat
    ⊢ LE.le (HMul.hMul (NNNorm.nnnorm (p.changeOriginSeries k n)) (HPow.hPow r n)) …
  -/
  simp only [NNReal.tsum_mul_right]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : NNReal
    hr : LT.lt (↑r) p.radius
    k n : Nat
    ⊢ LE.le (HMul.hMul (NNNorm.nnnorm (p.changeOriginSeries k n)) (HPow.hPow r n)) …
  -/
  exact mul_le_mul' (p.nnnorm_changeOriginSeries_le_tsum _ _) le_rfl
  /-
    🎉 no goals
  -/


theorem le_changeOriginSeries_radius (k : ℕ) : p.radius ≤ (p.changeOriginSeries k).radius :=
  ENNReal.le_of_forall_nnreal_lt fun _r hr =>
    le_radius_of_summable_nnnorm _ (p.changeOriginSeries_summable_aux₃ hr k)


theorem nnnorm_changeOrigin_le (k : ℕ) (h : (‖x‖₊ : ℝ≥0∞) < p.radius) :
    ‖p.changeOrigin x k‖₊ ≤
      ∑' s : Σl : ℕ, { s : Finset (Fin (k + l)) // s.card = l }, ‖p (k + s.1)‖₊ * ‖x‖₊ ^ s.1 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    k : Nat
    h : LT.lt (↑(NNNorm.nnnorm x)) p.radius
    ⊢ LE.le (NNNorm.nnnorm (p.changeOrigin x k)) (tsum fun s => HMul.hMul (NNNorm. …
  -/
  refine tsum_of_nnnorm_bounded ?_ fun l => p.nnnorm_changeOriginSeries_apply_le_tsum k l x
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    k : Nat
    h : LT.lt (↑(NNNorm.nnnorm x)) p.radius
    ⊢ HasSum (fun l => tsum fun x_1 => HMul.hMul (NNNorm.nnnorm (p (HAdd.hAdd k l) …
  -/
  have := p.changeOriginSeries_summable_aux₂ h k
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    k : Nat
    h : LT.lt (↑(NNNorm.nnnorm x)) p.radius
    this : Summable fun s => HMul.hMul (NNNorm.nnnorm (p (HAdd.hAdd k s.fst))) (HP …
    ⊢ HasSum (fun l => tsum fun x_1 => HMul.hMul (NNNorm.nnnorm (p (HAdd.hAdd k l) …
  -/
  refine HasSum.sigma this.hasSum fun l => ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    k : Nat
    h : LT.lt (↑(NNNorm.nnnorm x)) p.radius
    this : Summable fun s => HMul.hMul (NNNorm.nnnorm (p (HAdd.hAdd k s.fst))) (HP …
    l : Nat
    ⊢ HasSum (fun c => HMul.hMul (NNNorm.nnnorm (p (HAdd.hAdd k ⟨l, c⟩.fst))) (HPo …
  -/
  exact ((NNReal.summable_sigma.1 this).1 l).hasSum
  /-
    🎉 no goals
  -/


/-- The radius of convergence of `p.changeOrigin x` is at least `p.radius - ‖x‖`. In other words,
`p.changeOrigin x` is well defined on the largest ball contained in the original ball of
convergence. -/
theorem changeOrigin_radius : p.radius - ‖x‖₊ ≤ (p.changeOrigin x).radius := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    ⊢ LE.le (HSub.hSub p.radius ↑(NNNorm.nnnorm x)) (p.changeOrigin x).radius
  -/
  refine ENNReal.le_of_forall_pos_nnreal_lt fun r _h0 hr => ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    r : NNReal
    _h0 : LT.lt 0 r
    hr : LT.lt (↑r) (HSub.hSub p.radius ↑(NNNorm.nnnorm x))
    ⊢ LE.le (↑r) (p.changeOrigin x).radius
  -/
  rw [lt_tsub_iff_right, add_comm] at hr
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    r : NNReal
    _h0 : LT.lt 0 r
    hr : LT.lt (HAdd.hAdd ↑(NNNorm.nnnorm x) ↑r) p.radius
    ⊢ LE.le (↑r) (p.changeOrigin x).radius
  -/
  have hr' : (‖x‖₊ : ℝ≥0∞) < p.radius := (le_add_right le_rfl).trans_lt hr
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    r : NNReal
    _h0 : LT.lt 0 r
    hr : LT.lt (HAdd.hAdd ↑(NNNorm.nnnorm x) ↑r) p.radius
    hr' : LT.lt (↑(NNNorm.nnnorm x)) p.radius
    ⊢ LE.le (↑r) (p.changeOrigin x).radius
  -/
  apply le_radius_of_summable_nnnorm
  have : ∀ k : ℕ,
      ‖p.changeOrigin x k‖₊ * r ^ k ≤
        (∑' s : Σl : ℕ, { s : Finset (Fin (k + l)) // s.card = l }, ‖p (k + s.1)‖₊ * ‖x‖₊ ^ s.1) *
          r ^ k :=
    fun k => mul_le_mul_right' (p.nnnorm_changeOrigin_le k hr') (r ^ k)
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    r : NNReal
    _h0 : LT.lt 0 r
    hr : LT.lt (HAdd.hAdd ↑(NNNorm.nnnorm x) ↑r) p.radius
    hr' : LT.lt (↑(NNNorm.nnnorm x)) p.radius
    this : ∀ (k : Nat), LE.le (HMul.hMul (NNNorm.nnnorm (p.changeOrigin x k)) (HPo …
    ⊢ Summable fun n => HMul.hMul (NNNorm.nnnorm (p.changeOrigin x n)) (HPow.hPow  …
  -/
  refine NNReal.summable_of_le this ?_
  simpa only [← NNReal.tsum_mul_right] using
    (NNReal.summable_sigma.1 (p.changeOriginSeries_summable_aux₁ hr)).2


/-- `derivSeries p` is a power series for `fderiv 𝕜 f` if `p` is a power series for `f`,
see `HasFPowerSeriesOnBall.fderiv`. -/
def derivSeries : FormalMultilinearSeries 𝕜 E (E →L[𝕜] F) :=
  (continuousMultilinearCurryFin1 𝕜 E F : (E[×1]→L[𝕜] F) →L[𝕜] E →L[𝕜] F)
    |>.compFormalMultilinearSeries (p.changeOriginSeries 1)


theorem radius_le_radius_derivSeries : p.radius ≤ p.derivSeries.radius := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    ⊢ LE.le p.radius p.derivSeries.radius
  -/
  apply (p.le_changeOriginSeries_radius 1).trans (radius_le_of_le (fun n ↦ ?_))
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    ⊢ LE.le (Norm.norm (p.derivSeries n)) (Norm.norm (p.changeOriginSeries 1 n))
  -/
  apply (ContinuousLinearMap.norm_compContinuousMultilinearMap_le _ _).trans
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    ⊢ LE.le (HMul.hMul (Norm.norm ↑{ toLinearEquiv := (continuousMultilinearCurryF …
  -/
  apply mul_le_of_le_one_left (norm_nonneg  _)
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    ⊢ LE.le (Norm.norm ↑{ toLinearEquiv := (continuousMultilinearCurryFin1 𝕜 E F). …
  -/
  exact ContinuousLinearMap.opNorm_le_bound _ zero_le_one (by simp)
  /-
    🎉 no goals
  -/


theorem derivSeries_eq_zero {n : ℕ} (hp : p (n + 1) = 0) : p.derivSeries n = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hp : Eq (p (HAdd.hAdd n 1)) 0
    ⊢ Eq (p.derivSeries n) 0
  -/
  suffices p.changeOriginSeries 1 n = 0 by ext v; simp [derivSeries, this]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hp : Eq (p (HAdd.hAdd n 1)) 0
    ⊢ Eq (p.changeOriginSeries 1 n) 0
  -/
  apply Finset.sum_eq_zero (fun s hs ↦ ?_)
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hp : Eq (p (HAdd.hAdd n 1)) 0
    s : Subtype fun s => Eq s.card n
    hs : Membership.mem Finset.univ s
    ⊢ Eq (p.changeOriginSeriesTerm 1 n ↑s ⋯) 0
  -/
  ext v
  /-
    case H.H
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hp : Eq (p (HAdd.hAdd n 1)) 0
    s : Subtype fun s => Eq s.card n
    hs : Membership.mem Finset.univ s
    v : Fin n → E
    x✝ : Fin 1 → E
    ⊢ Eq (((p.changeOriginSeriesTerm 1 n ↑s ⋯) v) x✝) ((0 v) x✝)
  -/
  have : p (1 + n) = 0 := p.congr_zero (by abel) hp
  simp [changeOriginSeriesTerm, ContinuousMultilinearMap.curryFinFinset_apply,
    ContinuousMultilinearMap.zero_apply, this]


theorem hasFPowerSeriesOnBall_changeOrigin (k : ℕ) (hr : 0 < p.radius) :
    HasFPowerSeriesOnBall (fun x => p.changeOrigin x k) (p.changeOriginSeries k) 0 p.radius :=
  have := p.le_changeOriginSeries_radius k
  ((p.changeOriginSeries k).hasFPowerSeriesOnBall (hr.trans_le this)).mono hr this


/-- Summing the series `p.changeOrigin x` at a point `y` gives back `p (x + y)`. -/
theorem changeOrigin_eval (h : (‖x‖₊ + ‖y‖₊ : ℝ≥0∞) < p.radius) :
    (p.changeOrigin x).sum y = p.sum (x + y) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    p : FormalMultilinearSeries 𝕜 E F
    x y : E
    h : LT.lt (HAdd.hAdd ↑(NNNorm.nnnorm x) ↑(NNNorm.nnnorm y)) p.radius
    ⊢ Eq ((p.changeOrigin x).sum y) (p.sum (HAdd.hAdd x y))
  -/
  have radius_pos : 0 < p.radius := lt_of_le_of_lt (zero_le _) h
  have x_mem_ball : x ∈ EMetric.ball (0 : E) p.radius :=
    mem_emetric_ball_zero_iff.2 ((le_add_right le_rfl).trans_lt h)
  have y_mem_ball : y ∈ EMetric.ball (0 : E) (p.changeOrigin x).radius := by
    refine mem_emetric_ball_zero_iff.2 (lt_of_lt_of_le ?_ p.changeOrigin_radius)
    rwa [lt_tsub_iff_right, add_comm]
  have x_add_y_mem_ball : x + y ∈ EMetric.ball (0 : E) p.radius := by
    refine mem_emetric_ball_zero_iff.2 (lt_of_le_of_lt ?_ h)
    exact mod_cast nnnorm_add_le x y
  set f : (Σ k l : ℕ, { s : Finset (Fin (k + l)) // s.card = l }) → F := fun s =>
    p.changeOriginSeriesTerm s.1 s.2.1 s.2.2 s.2.2.2 (fun _ => x) fun _ => y
  have hsf : Summable f := by
    refine .of_nnnorm_bounded _ (p.changeOriginSeries_summable_aux₁ h) ?_
    rintro ⟨k, l, s, hs⟩
    dsimp only [Subtype.coe_mk]
    exact p.nnnorm_changeOriginSeriesTerm_apply_le _ _ _ _ _ _
  have hf : HasSum f ((p.changeOrigin x).sum y) := by
    refine HasSum.sigma_of_hasSum ((p.changeOrigin x).summable y_mem_ball).hasSum (fun k => ?_) hsf
    · dsimp only [f]
      refine ContinuousMultilinearMap.hasSum_eval ?_ _
      have := (p.hasFPowerSeriesOnBall_changeOrigin k radius_pos).hasSum x_mem_ball
      rw [zero_add] at this
      refine HasSum.sigma_of_hasSum this (fun l => ?_) ?_
      · simp only [changeOriginSeries, ContinuousMultilinearMap.sum_apply]
        apply hasSum_fintype
      · refine .of_nnnorm_bounded _
          (p.changeOriginSeries_summable_aux₂ (mem_emetric_ball_zero_iff.1 x_mem_ball) k)
            fun s => ?_
        refine (ContinuousMultilinearMap.le_opNNNorm _ _).trans_eq ?_
        simp
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    p : FormalMultilinearSeries 𝕜 E F
    x y : E
    h : LT.lt (HAdd.hAdd ↑(NNNorm.nnnorm x) ↑(NNNorm.nnnorm y)) p.radius
    radius_pos : LT.lt 0 p.radius
    x_mem_ball : Membership.mem (EMetric.ball 0 p.radius) x
    y_mem_ball : Membership.mem (EMetric.ball 0 (p.changeOrigin x).radius) y
    x_add_y_mem_ball : Membership.mem (EMetric.ball 0 p.radius) (HAdd.hAdd x y)
    f : (Sigma fun k => Sigma fun l => Subtype fun s => Eq s.card l) → F := fun s  …
    hsf : Summable f
    hf : HasSum f ((p.changeOrigin x).sum y)
    ⊢ Eq ((p.changeOrigin x).sum y) (p.sum (HAdd.hAdd x y))
  -/
  refine hf.unique (changeOriginIndexEquiv.symm.hasSum_iff.1 ?_)
  refine HasSum.sigma_of_hasSum
    (p.hasSum x_add_y_mem_ball) (fun n => ?_) (changeOriginIndexEquiv.symm.summable_iff.2 hsf)
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    p : FormalMultilinearSeries 𝕜 E F
    x y : E
    h : LT.lt (HAdd.hAdd ↑(NNNorm.nnnorm x) ↑(NNNorm.nnnorm y)) p.radius
    radius_pos : LT.lt 0 p.radius
    x_mem_ball : Membership.mem (EMetric.ball 0 p.radius) x
    y_mem_ball : Membership.mem (EMetric.ball 0 (p.changeOrigin x).radius) y
    x_add_y_mem_ball : Membership.mem (EMetric.ball 0 p.radius) (HAdd.hAdd x y)
    f : (Sigma fun k => Sigma fun l => Subtype fun s => Eq s.card l) → F := fun s  …
    hsf : Summable f
    hf : HasSum f ((p.changeOrigin x).sum y)
    n : Nat
    ⊢ HasSum (fun c => Function.comp f ⇑FormalMultilinearSeries.changeOriginIndexE …
  -/
  erw [(p n).map_add_univ (fun _ => x) fun _ => y]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    p : FormalMultilinearSeries 𝕜 E F
    x y : E
    h : LT.lt (HAdd.hAdd ↑(NNNorm.nnnorm x) ↑(NNNorm.nnnorm y)) p.radius
    radius_pos : LT.lt 0 p.radius
    x_mem_ball : Membership.mem (EMetric.ball 0 p.radius) x
    y_mem_ball : Membership.mem (EMetric.ball 0 (p.changeOrigin x).radius) y
    x_add_y_mem_ball : Membership.mem (EMetric.ball 0 p.radius) (HAdd.hAdd x y)
    f : (Sigma fun k => Sigma fun l => Subtype fun s => Eq s.card l) → F := fun s  …
    hsf : Summable f
    hf : HasSum f ((p.changeOrigin x).sum y)
    n : Nat
    ⊢ HasSum (fun c => Function.comp f ⇑FormalMultilinearSeries.changeOriginIndexE …
  -/
  simp_rw [← changeOriginSeriesTerm_changeOriginIndexEquiv_symm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    p : FormalMultilinearSeries 𝕜 E F
    x y : E
    h : LT.lt (HAdd.hAdd ↑(NNNorm.nnnorm x) ↑(NNNorm.nnnorm y)) p.radius
    radius_pos : LT.lt 0 p.radius
    x_mem_ball : Membership.mem (EMetric.ball 0 p.radius) x
    y_mem_ball : Membership.mem (EMetric.ball 0 (p.changeOrigin x).radius) y
    x_add_y_mem_ball : Membership.mem (EMetric.ball 0 p.radius) (HAdd.hAdd x y)
    f : (Sigma fun k => Sigma fun l => Subtype fun s => Eq s.card l) → F := fun s  …
    hsf : Summable f
    hf : HasSum f ((p.changeOrigin x).sum y)
    n : Nat
    ⊢ HasSum (fun c => Function.comp f ⇑FormalMultilinearSeries.changeOriginIndexE …
  -/
  exact hasSum_fintype (fun c => f (changeOriginIndexEquiv.symm ⟨n, c⟩))
  /-
    🎉 no goals
  -/


/-- Power series terms are analytic as we vary the origin -/
theorem analyticAt_changeOrigin (p : FormalMultilinearSeries 𝕜 E F) (rp : p.radius > 0) (n : ℕ) :
    AnalyticAt 𝕜 (fun x ↦ p.changeOrigin x n) 0 :=
  (FormalMultilinearSeries.hasFPowerSeriesOnBall_changeOrigin p n rp).analyticAt


/-- If a function admits a power series expansion `p` within a set `s` on a ball `B (x, r)`, then
it also admits a power series on any subball of this ball (even with a different center provided
it belongs to `s`), given by `p.changeOrigin`. -/
theorem HasFPowerSeriesWithinOnBall.changeOrigin (hf : HasFPowerSeriesWithinOnBall f p s x r)
    (h : (‖y‖₊ : ℝ≥0∞) < r) (hy : x + y ∈ insert x s) :
    HasFPowerSeriesWithinOnBall f (p.changeOrigin y) s (x + y) (r - ‖y‖₊) where
  r_le := by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x y : E
      r : ENNReal
      hf : HasFPowerSeriesWithinOnBall f p s x r
      h : LT.lt (↑(NNNorm.nnnorm y)) r
      hy : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
      ⊢ LE.le (HSub.hSub r ↑(NNNorm.nnnorm y)) (p.changeOrigin y).radius
    -/
    apply le_trans _ p.changeOrigin_radius
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x y : E
      r : ENNReal
      hf : HasFPowerSeriesWithinOnBall f p s x r
      h : LT.lt (↑(NNNorm.nnnorm y)) r
      hy : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
      ⊢ LE.le (HSub.hSub r ↑(NNNorm.nnnorm y)) (HSub.hSub p.radius ↑(NNNorm.nnnorm y))
    -/
    exact tsub_le_tsub hf.r_le le_rfl
    /-
      🎉 no goals
    -/
              /-
                𝕜 : Type u_1
                E : Type u_2
                F : Type u_3
                inst✝⁵ : NontriviallyNormedField 𝕜
                inst✝⁴ : NormedAddCommGroup E
                inst✝³ : NormedSpace 𝕜 E
                inst✝² : NormedAddCommGroup F
                inst✝¹ : NormedSpace 𝕜 F
                inst✝ : CompleteSpace F
                f : E → F
                p : FormalMultilinearSeries 𝕜 E F
                s : Set E
                x y : E
                r : ENNReal
                hf : HasFPowerSeriesWithinOnBall f p s x r
                h : LT.lt (↑(NNNorm.nnnorm y)) r
                hy : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
                ⊢ LT.lt 0 (HSub.hSub r ↑(NNNorm.nnnorm y))
              -/
  r_pos := by simp [h]
              /-
                🎉 no goals
              -/
  hasSum {z} h'z hz := by
    have : f (x + y + z) =
        FormalMultilinearSeries.sum (FormalMultilinearSeries.changeOrigin p y) z := by
      rw [mem_emetric_ball_zero_iff, lt_tsub_iff_right, add_comm] at hz
      rw [p.changeOrigin_eval (hz.trans_le hf.r_le), add_assoc, hf.sum]
      · have : insert (x + y) s ⊆ insert (x + y) (insert x s) := by
          apply insert_subset_insert (subset_insert _ _)
        rw [insert_eq_of_mem hy] at this
        apply this
        simpa [add_assoc] using h'z
      refine mem_emetric_ball_zero_iff.2 (lt_of_le_of_lt ?_ hz)
      exact mod_cast nnnorm_add_le y z
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x y : E
      r : ENNReal
      hf : HasFPowerSeriesWithinOnBall f p s x r
      h : LT.lt (↑(NNNorm.nnnorm y)) r
      hy : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
      z : E
      h'z : Membership.mem (Insert.insert (HAdd.hAdd x y) s) (HAdd.hAdd (HAdd.hAdd x …
      hz : Membership.mem (EMetric.ball 0 (HSub.hSub r ↑(NNNorm.nnnorm y))) z
      this : Eq (f (HAdd.hAdd (HAdd.hAdd x y) z)) ((p.changeOrigin y).sum z)
      ⊢ HasSum (fun n => (p.changeOrigin y n) fun x => z) (f (HAdd.hAdd (HAdd.hAdd x …
    -/
    rw [this]
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x y : E
      r : ENNReal
      hf : HasFPowerSeriesWithinOnBall f p s x r
      h : LT.lt (↑(NNNorm.nnnorm y)) r
      hy : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
      z : E
      h'z : Membership.mem (Insert.insert (HAdd.hAdd x y) s) (HAdd.hAdd (HAdd.hAdd x …
      hz : Membership.mem (EMetric.ball 0 (HSub.hSub r ↑(NNNorm.nnnorm y))) z
      this : Eq (f (HAdd.hAdd (HAdd.hAdd x y) z)) ((p.changeOrigin y).sum z)
      ⊢ HasSum (fun n => (p.changeOrigin y n) fun x => z) ((p.changeOrigin y).sum z)
    -/
    apply (p.changeOrigin y).hasSum
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x y : E
      r : ENNReal
      hf : HasFPowerSeriesWithinOnBall f p s x r
      h : LT.lt (↑(NNNorm.nnnorm y)) r
      hy : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
      z : E
      h'z : Membership.mem (Insert.insert (HAdd.hAdd x y) s) (HAdd.hAdd (HAdd.hAdd x …
      hz : Membership.mem (EMetric.ball 0 (HSub.hSub r ↑(NNNorm.nnnorm y))) z
      this : Eq (f (HAdd.hAdd (HAdd.hAdd x y) z)) ((p.changeOrigin y).sum z)
      ⊢ Membership.mem (EMetric.ball 0 (p.changeOrigin y).radius) z
    -/
    refine EMetric.ball_subset_ball (le_trans ?_ p.changeOrigin_radius) hz
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x y : E
      r : ENNReal
      hf : HasFPowerSeriesWithinOnBall f p s x r
      h : LT.lt (↑(NNNorm.nnnorm y)) r
      hy : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
      z : E
      h'z : Membership.mem (Insert.insert (HAdd.hAdd x y) s) (HAdd.hAdd (HAdd.hAdd x …
      hz : Membership.mem (EMetric.ball 0 (HSub.hSub r ↑(NNNorm.nnnorm y))) z
      this : Eq (f (HAdd.hAdd (HAdd.hAdd x y) z)) ((p.changeOrigin y).sum z)
      ⊢ LE.le (HSub.hSub r ↑(NNNorm.nnnorm y)) (HSub.hSub p.radius ↑(NNNorm.nnnorm y))
    -/
    exact tsub_le_tsub hf.r_le le_rfl
    /-
      🎉 no goals
    -/


/-- If a function admits a power series expansion `p` on a ball `B (x, r)`, then it also admits a
power series on any subball of this ball (even with a different center), given by `p.changeOrigin`.
-/
theorem HasFPowerSeriesOnBall.changeOrigin (hf : HasFPowerSeriesOnBall f p x r)
    (h : (‖y‖₊ : ℝ≥0∞) < r) : HasFPowerSeriesOnBall f (p.changeOrigin y) (x + y) (r - ‖y‖₊) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    x y : E
    r : ENNReal
    hf : HasFPowerSeriesOnBall f p x r
    h : LT.lt (↑(NNNorm.nnnorm y)) r
    ⊢ HasFPowerSeriesOnBall f (p.changeOrigin y) (HAdd.hAdd x y) (HSub.hSub r ↑(NN …
  -/
  rw [← hasFPowerSeriesWithinOnBall_univ] at hf ⊢
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    x y : E
    r : ENNReal
    hf : HasFPowerSeriesWithinOnBall f p Set.univ x r
    h : LT.lt (↑(NNNorm.nnnorm y)) r
    ⊢ HasFPowerSeriesWithinOnBall f (p.changeOrigin y) Set.univ (HAdd.hAdd x y) (H …
  -/
  exact hf.changeOrigin h (by simp)
  /-
    🎉 no goals
  -/


/-- If a function admits a power series expansion `p` on an open ball `B (x, r)`, then
it is analytic at every point of this ball. -/
theorem HasFPowerSeriesWithinOnBall.analyticWithinAt_of_mem
    (hf : HasFPowerSeriesWithinOnBall f p s x r)
    (h : y ∈ insert x s ∩ EMetric.ball x r) : AnalyticWithinAt 𝕜 f s y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x y : E
    r : ENNReal
    hf : HasFPowerSeriesWithinOnBall f p s x r
    h : Membership.mem (Inter.inter (Insert.insert x s) (EMetric.ball x r)) y
    ⊢ AnalyticWithinAt 𝕜 f s y
  -/
  have : (‖y - x‖₊ : ℝ≥0∞) < r := by simpa [edist_eq_coe_nnnorm_sub] using h.2
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x y : E
    r : ENNReal
    hf : HasFPowerSeriesWithinOnBall f p s x r
    h : Membership.mem (Inter.inter (Insert.insert x s) (EMetric.ball x r)) y
    this : LT.lt (↑(NNNorm.nnnorm (HSub.hSub y x))) r
    ⊢ AnalyticWithinAt 𝕜 f s y
  -/
  have := hf.changeOrigin this (by simpa using h.1)
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x y : E
    r : ENNReal
    hf : HasFPowerSeriesWithinOnBall f p s x r
    h : Membership.mem (Inter.inter (Insert.insert x s) (EMetric.ball x r)) y
    this✝ : LT.lt (↑(NNNorm.nnnorm (HSub.hSub y x))) r
    this : HasFPowerSeriesWithinOnBall f (p.changeOrigin (HSub.hSub y x)) s (HAdd. …
    ⊢ AnalyticWithinAt 𝕜 f s y
  -/
  rw [add_sub_cancel] at this
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x y : E
    r : ENNReal
    hf : HasFPowerSeriesWithinOnBall f p s x r
    h : Membership.mem (Inter.inter (Insert.insert x s) (EMetric.ball x r)) y
    this✝ : LT.lt (↑(NNNorm.nnnorm (HSub.hSub y x))) r
    this : HasFPowerSeriesWithinOnBall f (p.changeOrigin (HSub.hSub y x)) s y (HSu …
    ⊢ AnalyticWithinAt 𝕜 f s y
  -/
  exact this.analyticWithinAt
  /-
    🎉 no goals
  -/


/-- If a function admits a power series expansion `p` on an open ball `B (x, r)`, then
it is analytic at every point of this ball. -/
theorem HasFPowerSeriesOnBall.analyticAt_of_mem (hf : HasFPowerSeriesOnBall f p x r)
    (h : y ∈ EMetric.ball x r) : AnalyticAt 𝕜 f y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    x y : E
    r : ENNReal
    hf : HasFPowerSeriesOnBall f p x r
    h : Membership.mem (EMetric.ball x r) y
    ⊢ AnalyticAt 𝕜 f y
  -/
  rw [← hasFPowerSeriesWithinOnBall_univ] at hf
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    x y : E
    r : ENNReal
    hf : HasFPowerSeriesWithinOnBall f p Set.univ x r
    h : Membership.mem (EMetric.ball x r) y
    ⊢ AnalyticAt 𝕜 f y
  -/
  rw [← analyticWithinAt_univ]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    x y : E
    r : ENNReal
    hf : HasFPowerSeriesWithinOnBall f p Set.univ x r
    h : Membership.mem (EMetric.ball x r) y
    ⊢ AnalyticWithinAt 𝕜 f Set.univ y
  -/
  exact hf.analyticWithinAt_of_mem (by simpa using h)
  /-
    🎉 no goals
  -/


theorem HasFPowerSeriesWithinOnBall.analyticOn (hf : HasFPowerSeriesWithinOnBall f p s x r) :
    AnalyticOn 𝕜 f (insert x s ∩ EMetric.ball x r) :=
  fun _ hy ↦ ((analyticWithinAt_insert (y := x)).2 (hf.analyticWithinAt_of_mem hy)).mono
    inter_subset_left


theorem HasFPowerSeriesOnBall.analyticOnNhd (hf : HasFPowerSeriesOnBall f p x r) :
    AnalyticOnNhd 𝕜 f (EMetric.ball x r) :=
  fun _y hy => hf.analyticAt_of_mem hy


@[deprecated (since := "2024-09-26")]
alias HasFPowerSeriesOnBall.analyticOn := HasFPowerSeriesOnBall.analyticOnNhd


variable (𝕜 f) in
/-- For any function `f` from a normed vector space to a Banach space, the set of points `x` such
that `f` is analytic at `x` is open. -/
theorem isOpen_analyticAt : IsOpen { x | AnalyticAt 𝕜 f x } := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    ⊢ IsOpen (setOf fun x => AnalyticAt 𝕜 f x)
  -/
  rw [isOpen_iff_mem_nhds]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    ⊢ ∀ (x : E), Membership.mem (setOf fun x => AnalyticAt 𝕜 f x) x → Membership.m …
  -/
  rintro x ⟨p, r, hr⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    x : E
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    hr : HasFPowerSeriesOnBall f p x r
    ⊢ Membership.mem (nhds x) (setOf fun x => AnalyticAt 𝕜 f x)
  -/
  exact mem_of_superset (EMetric.ball_mem_nhds _ hr.r_pos) fun y hy => hr.analyticAt_of_mem hy
  /-
    🎉 no goals
  -/


theorem AnalyticAt.eventually_analyticAt (h : AnalyticAt 𝕜 f x) :
    ∀ᶠ y in 𝓝 x, AnalyticAt 𝕜 f y :=
  (isOpen_analyticAt 𝕜 f).mem_nhds h


theorem AnalyticAt.exists_mem_nhds_analyticOnNhd (h : AnalyticAt 𝕜 f x) :
    ∃ s ∈ 𝓝 x, AnalyticOnNhd 𝕜 f s :=
  h.eventually_analyticAt.exists_mem


@[deprecated (since := "2024-09-26")]
alias AnalyticAt.exists_mem_nhds_analyticOn := AnalyticAt.exists_mem_nhds_analyticOnNhd


/-- If we're analytic at a point, we're analytic in a nonempty ball -/
theorem AnalyticAt.exists_ball_analyticOnNhd (h : AnalyticAt 𝕜 f x) :
    ∃ r : ℝ, 0 < r ∧ AnalyticOnNhd 𝕜 f (Metric.ball x r) :=
  Metric.isOpen_iff.mp (isOpen_analyticAt _ _) _ h


@[deprecated (since := "2024-09-26")]
alias AnalyticAt.exists_ball_analyticOn := AnalyticAt.exists_ball_analyticOnNhd



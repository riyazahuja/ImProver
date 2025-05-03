/-- This version of the Landau notation `IsBigOWith C l f g` where `f` and `g` are two functions on
a type `α` and `l` is a filter on `α`, means that eventually for `l`, `‖f‖` is bounded by `C * ‖g‖`.
In other words, `‖f‖ / ‖g‖` is eventually bounded by `C`, modulo division by zero issues that are
avoided by this definition. Probably you want to use `IsBigO` instead of this relation. -/
irreducible_def IsBigOWith (c : ℝ) (l : Filter α) (f : α → E) (g : α → F) : Prop :=
  ∀ᶠ x in l, ‖f x‖ ≤ c * ‖g x‖


/-- Definition of `IsBigOWith`. We record it in a lemma as `IsBigOWith` is irreducible. -/
                                                                                 /-
                                                                                   α : Type u_1
                                                                                   E : Type u_3
                                                                                   F : Type u_4
                                                                                   inst✝¹ : Norm E
                                                                                   inst✝ : Norm F
                                                                                   c : Real
                                                                                   f : α → E
                                                                                   g : α → F
                                                                                   l : Filter α
                                                                                   ⊢ Iff (Asymptotics.IsBigOWith c l f g) (Filter.Eventually (fun x => LE.le (Nor …
                                                                                 -/
theorem isBigOWith_iff : IsBigOWith c l f g ↔ ∀ᶠ x in l, ‖f x‖ ≤ c * ‖g x‖ := by rw [IsBigOWith_def]
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


alias ⟨IsBigOWith.bound, IsBigOWith.of_bound⟩ := isBigOWith_iff


/-- The Landau notation `f =O[l] g` where `f` and `g` are two functions on a type `α` and `l` is
a filter on `α`, means that eventually for `l`, `‖f‖` is bounded by a constant multiple of `‖g‖`.
In other words, `‖f‖ / ‖g‖` is eventually bounded, modulo division by zero issues that are avoided
by this definition. -/
irreducible_def IsBigO (l : Filter α) (f : α → E) (g : α → F) : Prop :=
  ∃ c : ℝ, IsBigOWith c l f g


@[inherit_doc]
notation:100 f " =O[" l "] " g:100 => IsBigO l f g


/-- Definition of `IsBigO` in terms of `IsBigOWith`. We record it in a lemma as `IsBigO` is
irreducible. -/
                                                                              /-
                                                                                α : Type u_1
                                                                                E : Type u_3
                                                                                F : Type u_4
                                                                                inst✝¹ : Norm E
                                                                                inst✝ : Norm F
                                                                                f : α → E
                                                                                g : α → F
                                                                                l : Filter α
                                                                                ⊢ Iff (Asymptotics.IsBigO l f g) (Exists fun c => Asymptotics.IsBigOWith c l f …
                                                                              -/
theorem isBigO_iff_isBigOWith : f =O[l] g ↔ ∃ c : ℝ, IsBigOWith c l f g := by rw [IsBigO_def]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


/-- Definition of `IsBigO` in terms of filters. -/
theorem isBigO_iff : f =O[l] g ↔ ∃ c : ℝ, ∀ᶠ x in l, ‖f x‖ ≤ c * ‖g x‖ := by
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    l : Filter α
    ⊢ Iff (Asymptotics.IsBigO l f g) (Exists fun c => Filter.Eventually (fun x =>  …
  -/
  simp only [IsBigO_def, IsBigOWith_def]
  /-
    🎉 no goals
  -/


/-- Definition of `IsBigO` in terms of filters, with a positive constant. -/
theorem isBigO_iff' {g : α → E'''} :
    f =O[l] g ↔ ∃ c > 0, ∀ᶠ x in l, ‖f x‖ ≤ c * ‖g x‖ := by
  /-
    α : Type u_1
    E : Type u_3
    E''' : Type u_12
    inst✝¹ : Norm E
    inst✝ : SeminormedAddGroup E'''
    f : α → E
    l : Filter α
    g : α → E'''
    ⊢ Iff (Asymptotics.IsBigO l f g) (Exists fun c => And (GT.gt c 0) (Filter.Even …
  -/
  refine ⟨fun h => ?mp, fun h => ?mpr⟩
  case mp =>
    rw [isBigO_iff] at h
    obtain ⟨c, hc⟩ := h
    refine ⟨max c 1, zero_lt_one.trans_le (le_max_right _ _), ?_⟩
    filter_upwards [hc] with x hx
    apply hx.trans
    gcongr
    exact le_max_left _ _
  case mpr =>
    rw [isBigO_iff]
    obtain ⟨c, ⟨_, hc⟩⟩ := h
    exact ⟨c, hc⟩


/-- Definition of `IsBigO` in terms of filters, with the constant in the lower bound. -/
theorem isBigO_iff'' {g : α → E'''} :
    f =O[l] g ↔ ∃ c > 0, ∀ᶠ x in l, c * ‖f x‖ ≤ ‖g x‖ := by
  /-
    α : Type u_1
    E : Type u_3
    E''' : Type u_12
    inst✝¹ : Norm E
    inst✝ : SeminormedAddGroup E'''
    f : α → E
    l : Filter α
    g : α → E'''
    ⊢ Iff (Asymptotics.IsBigO l f g) (Exists fun c => And (GT.gt c 0) (Filter.Even …
  -/
  refine ⟨fun h => ?mp, fun h => ?mpr⟩
  case mp =>
    rw [isBigO_iff'] at h
    obtain ⟨c, ⟨hc_pos, hc⟩⟩ := h
    refine ⟨c⁻¹, ⟨by positivity, ?_⟩⟩
    filter_upwards [hc] with x hx
    rwa [inv_mul_le_iff₀ (by positivity)]
  case mpr =>
    rw [isBigO_iff']
    obtain ⟨c, ⟨hc_pos, hc⟩⟩ := h
    refine ⟨c⁻¹, ⟨by positivity, ?_⟩⟩
    filter_upwards [hc] with x hx
    rwa [← inv_inv c, inv_mul_le_iff₀ (by positivity)] at hx


theorem IsBigO.of_bound (c : ℝ) (h : ∀ᶠ x in l, ‖f x‖ ≤ c * ‖g x‖) : f =O[l] g :=
  isBigO_iff.2 ⟨c, h⟩


theorem IsBigO.of_bound' (h : ∀ᶠ x in l, ‖f x‖ ≤ ‖g x‖) : f =O[l] g :=
  IsBigO.of_bound 1 <| by
    /-
      α : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝¹ : Norm E
      inst✝ : Norm F
      f : α → E
      g : α → F
      l : Filter α
      h : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (Norm.norm (g x))) l
      ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul 1 (Norm.norm  …
    -/
    simp_rw [one_mul]
    /-
      α : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝¹ : Norm E
      inst✝ : Norm F
      f : α → E
      g : α → F
      l : Filter α
      h : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (Norm.norm (g x))) l
      ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (Norm.norm (g x))) l
    -/
    exact h
    /-
      🎉 no goals
    -/


theorem IsBigO.bound : f =O[l] g → ∃ c : ℝ, ∀ᶠ x in l, ‖f x‖ ≤ c * ‖g x‖ :=
  isBigO_iff.1


/-- The Landau notation `f =o[l] g` where `f` and `g` are two functions on a type `α` and `l` is
a filter on `α`, means that eventually for `l`, `‖f‖` is bounded by an arbitrarily small constant
multiple of `‖g‖`. In other words, `‖f‖ / ‖g‖` tends to `0` along `l`, modulo division by zero
issues that are avoided by this definition. -/
irreducible_def IsLittleO (l : Filter α) (f : α → E) (g : α → F) : Prop :=
  ∀ ⦃c : ℝ⦄, 0 < c → IsBigOWith c l f g


@[inherit_doc]
notation:100 f " =o[" l "] " g:100 => IsLittleO l f g


/-- Definition of `IsLittleO` in terms of `IsBigOWith`. -/
theorem isLittleO_iff_forall_isBigOWith : f =o[l] g ↔ ∀ ⦃c : ℝ⦄, 0 < c → IsBigOWith c l f g := by
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    l : Filter α
    ⊢ Iff (Asymptotics.IsLittleO l f g) (∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsB …
  -/
  rw [IsLittleO_def]
  /-
    🎉 no goals
  -/


alias ⟨IsLittleO.forall_isBigOWith, IsLittleO.of_isBigOWith⟩ := isLittleO_iff_forall_isBigOWith


/-- Definition of `IsLittleO` in terms of filters. -/
theorem isLittleO_iff : f =o[l] g ↔ ∀ ⦃c : ℝ⦄, 0 < c → ∀ᶠ x in l, ‖f x‖ ≤ c * ‖g x‖ := by
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    l : Filter α
    ⊢ Iff (Asymptotics.IsLittleO l f g) (∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventual …
  -/
  simp only [IsLittleO_def, IsBigOWith_def]
  /-
    🎉 no goals
  -/


alias ⟨IsLittleO.bound, IsLittleO.of_bound⟩ := isLittleO_iff


theorem IsLittleO.def (h : f =o[l] g) (hc : 0 < c) : ∀ᶠ x in l, ‖f x‖ ≤ c * ‖g x‖ :=
  isLittleO_iff.1 h hc


theorem IsLittleO.def' (h : f =o[l] g) (hc : 0 < c) : IsBigOWith c l f g :=
  isBigOWith_iff.2 <| isLittleO_iff.1 h hc


theorem IsLittleO.eventuallyLE (h : f =o[l] g) : ∀ᶠ x in l, ‖f x‖ ≤ ‖g x‖ := by
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    l : Filter α
    h : Asymptotics.IsLittleO l f g
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (Norm.norm (g x))) l
  -/
  simpa using h.def zero_lt_one
  /-
    🎉 no goals
  -/


                                                                     /-
                                                                       α : Type u_1
                                                                       E : Type u_3
                                                                       F : Type u_4
                                                                       inst✝¹ : Norm E
                                                                       inst✝ : Norm F
                                                                       c : Real
                                                                       f : α → E
                                                                       g : α → F
                                                                       l : Filter α
                                                                       h : Asymptotics.IsBigOWith c l f g
                                                                       ⊢ Asymptotics.IsBigO l f g
                                                                     -/
theorem IsBigOWith.isBigO (h : IsBigOWith c l f g) : f =O[l] g := by rw [IsBigO_def]; exact ⟨c, h⟩
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


theorem IsLittleO.isBigOWith (hgf : f =o[l] g) : IsBigOWith 1 l f g :=
  hgf.def' zero_lt_one


theorem IsLittleO.isBigO (hgf : f =o[l] g) : f =O[l] g :=
  hgf.isBigOWith.isBigO


theorem IsBigO.isBigOWith : f =O[l] g → ∃ c : ℝ, IsBigOWith c l f g :=
  isBigO_iff_isBigOWith.1


theorem IsBigOWith.weaken (h : IsBigOWith c l f g') (hc : c ≤ c') : IsBigOWith c' l f g' :=
  IsBigOWith.of_bound <|
    mem_of_superset h.bound fun x hx =>
      calc
        ‖f x‖ ≤ c * ‖g' x‖ := hx
                    /-
                      α : Type u_1
                      E : Type u_3
                      F' : Type u_7
                      inst✝¹ : Norm E
                      inst✝ : SeminormedAddCommGroup F'
                      c c' : Real
                      f : α → E
                      g' : α → F'
                      l : Filter α
                      h : Asymptotics.IsBigOWith c l f g'
                      hc : LE.le c c'
                      x : α
                      hx : Membership.mem (setOf fun x => (fun x => LE.le (Norm.norm (f x)) (HMul.hM …
                      ⊢ LE.le (HMul.hMul c (Norm.norm (g' x))) (HMul.hMul c' (Norm.norm (g' x)))
                    -/
        _ ≤ _ := by gcongr
                    /-
                      🎉 no goals
                    -/


theorem IsBigOWith.exists_pos (h : IsBigOWith c l f g') :
    ∃ c' > 0, IsBigOWith c' l f g' :=
  ⟨max c 1, lt_of_lt_of_le zero_lt_one (le_max_right c 1), h.weaken <| le_max_left c 1⟩


theorem IsBigO.exists_pos (h : f =O[l] g') : ∃ c > 0, IsBigOWith c l f g' :=
  let ⟨_c, hc⟩ := h.isBigOWith
  hc.exists_pos


theorem IsBigOWith.exists_nonneg (h : IsBigOWith c l f g') :
    ∃ c' ≥ 0, IsBigOWith c' l f g' :=
  let ⟨c, cpos, hc⟩ := h.exists_pos
  ⟨c, le_of_lt cpos, hc⟩


theorem IsBigO.exists_nonneg (h : f =O[l] g') : ∃ c ≥ 0, IsBigOWith c l f g' :=
  let ⟨_c, hc⟩ := h.isBigOWith
  hc.exists_nonneg


/-- `f = O(g)` if and only if `IsBigOWith c f g` for all sufficiently large `c`. -/
theorem isBigO_iff_eventually_isBigOWith : f =O[l] g' ↔ ∀ᶠ c in atTop, IsBigOWith c l f g' :=
  isBigO_iff_isBigOWith.trans
    ⟨fun ⟨c, hc⟩ => mem_atTop_sets.2 ⟨c, fun _c' hc' => hc.weaken hc'⟩, fun h => h.exists⟩


/-- `f = O(g)` if and only if `∀ᶠ x in l, ‖f x‖ ≤ c * ‖g x‖` for all sufficiently large `c`. -/
theorem isBigO_iff_eventually : f =O[l] g' ↔ ∀ᶠ c in atTop, ∀ᶠ x in l, ‖f x‖ ≤ c * ‖g' x‖ :=
                                               /-
                                                 α : Type u_1
                                                 E : Type u_3
                                                 F' : Type u_7
                                                 inst✝¹ : Norm E
                                                 inst✝ : SeminormedAddCommGroup F'
                                                 f : α → E
                                                 g' : α → F'
                                                 l : Filter α
                                                 ⊢ Iff (Filter.Eventually (fun c => Asymptotics.IsBigOWith c l f g') Filter.atT …
                                               -/
  isBigO_iff_eventually_isBigOWith.trans <| by simp only [IsBigOWith_def]
                                               /-
                                                 🎉 no goals
                                               -/


theorem IsBigO.exists_mem_basis {ι} {p : ι → Prop} {s : ι → Set α} (h : f =O[l] g')
    (hb : l.HasBasis p s) :
    ∃ c > 0, ∃ i : ι, p i ∧ ∀ x ∈ s i, ‖f x‖ ≤ c * ‖g' x‖ :=
  flip Exists.imp h.exists_pos fun c h => by
    /-
      α : Type u_1
      E : Type u_3
      F' : Type u_7
      inst✝¹ : Norm E
      inst✝ : SeminormedAddCommGroup F'
      f : α → E
      g' : α → F'
      l : Filter α
      ι : Sort u_17
      p : ι → Prop
      s : ι → Set α
      h✝ : Asymptotics.IsBigO l f g'
      hb : l.HasBasis p s
      c : Real
      h : And (GT.gt c 0) (Asymptotics.IsBigOWith c l f g')
      ⊢ And (GT.gt c 0) (Exists fun i => And (p i) (∀ (x : α), Membership.mem (s i)  …
    -/
    simpa only [isBigOWith_iff, hb.eventually_iff, exists_prop] using h
    /-
      🎉 no goals
    -/


theorem isBigOWith_inv (hc : 0 < c) : IsBigOWith c⁻¹ l f g ↔ ∀ᶠ x in l, c * ‖f x‖ ≤ ‖g x‖ := by
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    c : Real
    f : α → E
    g : α → F
    l : Filter α
    hc : LT.lt 0 c
    ⊢ Iff (Asymptotics.IsBigOWith (Inv.inv c) l f g) (Filter.Eventually (fun x =>  …
  -/
  simp only [IsBigOWith_def, ← div_eq_inv_mul, le_div_iff₀' hc]
  /-
    🎉 no goals
  -/

-- We prove this lemma with strange assumptions to get two lemmas below automatically

theorem isLittleO_iff_nat_mul_le_aux (h₀ : (∀ x, 0 ≤ ‖f x‖) ∨ ∀ x, 0 ≤ ‖g x‖) :
    f =o[l] g ↔ ∀ n : ℕ, ∀ᶠ x in l, ↑n * ‖f x‖ ≤ ‖g x‖ := by
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    l : Filter α
    h₀ : Or (∀ (x : α), LE.le 0 (Norm.norm (f x))) (∀ (x : α), LE.le 0 (Norm.norm  …
    ⊢ Iff (Asymptotics.IsLittleO l f g) (∀ (n : Nat), Filter.Eventually (fun x =>  …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝¹ : Norm E
      inst✝ : Norm F
      f : α → E
      g : α → F
      l : Filter α
      h₀ : Or (∀ (x : α), LE.le 0 (Norm.norm (f x))) (∀ (x : α), LE.le 0 (Norm.norm  …
      ⊢ Asymptotics.IsLittleO l f g → ∀ (n : Nat), Filter.Eventually (fun x => LE.le …
    -/
  · rintro H (_ | n)
      /-
        case mp.zero
        α : Type u_1
        E : Type u_3
        F : Type u_4
        inst✝¹ : Norm E
        inst✝ : Norm F
        f : α → E
        g : α → F
        l : Filter α
        h₀ : Or (∀ (x : α), LE.le 0 (Norm.norm (f x))) (∀ (x : α), LE.le 0 (Norm.norm  …
        H : Asymptotics.IsLittleO l f g
        ⊢ Filter.Eventually (fun x => LE.le (HMul.hMul (↑0) (Norm.norm (f x))) (Norm.n …
      -/
    · refine (H.def one_pos).mono fun x h₀' => ?_
      /-
        case mp.zero
        α : Type u_1
        E : Type u_3
        F : Type u_4
        inst✝¹ : Norm E
        inst✝ : Norm F
        f : α → E
        g : α → F
        l : Filter α
        h₀ : Or (∀ (x : α), LE.le 0 (Norm.norm (f x))) (∀ (x : α), LE.le 0 (Norm.norm  …
        H : Asymptotics.IsLittleO l f g
        x : α
        h₀' : LE.le (Norm.norm (f x)) (HMul.hMul 1 (Norm.norm (g x)))
        ⊢ LE.le (HMul.hMul (↑0) (Norm.norm (f x))) (Norm.norm (g x))
      -/
      rw [Nat.cast_zero, zero_mul]
      /-
        case mp.zero
        α : Type u_1
        E : Type u_3
        F : Type u_4
        inst✝¹ : Norm E
        inst✝ : Norm F
        f : α → E
        g : α → F
        l : Filter α
        h₀ : Or (∀ (x : α), LE.le 0 (Norm.norm (f x))) (∀ (x : α), LE.le 0 (Norm.norm  …
        H : Asymptotics.IsLittleO l f g
        x : α
        h₀' : LE.le (Norm.norm (f x)) (HMul.hMul 1 (Norm.norm (g x)))
        ⊢ LE.le 0 (Norm.norm (g x))
      -/
      refine h₀.elim (fun hf => (hf x).trans ?_) fun hg => hg x
      /-
        case mp.zero
        α : Type u_1
        E : Type u_3
        F : Type u_4
        inst✝¹ : Norm E
        inst✝ : Norm F
        f : α → E
        g : α → F
        l : Filter α
        h₀ : Or (∀ (x : α), LE.le 0 (Norm.norm (f x))) (∀ (x : α), LE.le 0 (Norm.norm  …
        H : Asymptotics.IsLittleO l f g
        x : α
        h₀' : LE.le (Norm.norm (f x)) (HMul.hMul 1 (Norm.norm (g x)))
        hf : ∀ (x : α), LE.le 0 (Norm.norm (f x))
        ⊢ LE.le (Norm.norm (f x)) (Norm.norm (g x))
      -/
      rwa [one_mul] at h₀'
      /-
        🎉 no goals
      -/
      /-
        case mp.succ
        α : Type u_1
        E : Type u_3
        F : Type u_4
        inst✝¹ : Norm E
        inst✝ : Norm F
        f : α → E
        g : α → F
        l : Filter α
        h₀ : Or (∀ (x : α), LE.le 0 (Norm.norm (f x))) (∀ (x : α), LE.le 0 (Norm.norm  …
        H : Asymptotics.IsLittleO l f g
        n : Nat
        ⊢ Filter.Eventually (fun x => LE.le (HMul.hMul (↑(HAdd.hAdd n 1)) (Norm.norm ( …
      -/
    · have : (0 : ℝ) < n.succ := Nat.cast_pos.2 n.succ_pos
      /-
        case mp.succ
        α : Type u_1
        E : Type u_3
        F : Type u_4
        inst✝¹ : Norm E
        inst✝ : Norm F
        f : α → E
        g : α → F
        l : Filter α
        h₀ : Or (∀ (x : α), LE.le 0 (Norm.norm (f x))) (∀ (x : α), LE.le 0 (Norm.norm  …
        H : Asymptotics.IsLittleO l f g
        n : Nat
        this : LT.lt 0 ↑n.succ
        ⊢ Filter.Eventually (fun x => LE.le (HMul.hMul (↑(HAdd.hAdd n 1)) (Norm.norm ( …
      -/
      exact (isBigOWith_inv this).1 (H.def' <| inv_pos.2 this)
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝¹ : Norm E
      inst✝ : Norm F
      f : α → E
      g : α → F
      l : Filter α
      h₀ : Or (∀ (x : α), LE.le 0 (Norm.norm (f x))) (∀ (x : α), LE.le 0 (Norm.norm  …
      ⊢ (∀ (n : Nat), Filter.Eventually (fun x => LE.le (HMul.hMul (↑n) (Norm.norm ( …
    -/
  · refine fun H => isLittleO_iff.2 fun ε ε0 => ?_
    /-
      case mpr
      α : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝¹ : Norm E
      inst✝ : Norm F
      f : α → E
      g : α → F
      l : Filter α
      h₀ : Or (∀ (x : α), LE.le 0 (Norm.norm (f x))) (∀ (x : α), LE.le 0 (Norm.norm  …
      H : ∀ (n : Nat), Filter.Eventually (fun x => LE.le (HMul.hMul (↑n) (Norm.norm  …
      ε : Real
      ε0 : LT.lt 0 ε
      ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul ε (Norm.norm  …
    -/
    rcases exists_nat_gt ε⁻¹ with ⟨n, hn⟩
    /-
      case mpr.intro
      α : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝¹ : Norm E
      inst✝ : Norm F
      f : α → E
      g : α → F
      l : Filter α
      h₀ : Or (∀ (x : α), LE.le 0 (Norm.norm (f x))) (∀ (x : α), LE.le 0 (Norm.norm  …
      H : ∀ (n : Nat), Filter.Eventually (fun x => LE.le (HMul.hMul (↑n) (Norm.norm  …
      ε : Real
      ε0 : LT.lt 0 ε
      n : Nat
      hn : LT.lt (Inv.inv ε) ↑n
      ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul ε (Norm.norm  …
    -/
    have hn₀ : (0 : ℝ) < n := (inv_pos.2 ε0).trans hn
    /-
      case mpr.intro
      α : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝¹ : Norm E
      inst✝ : Norm F
      f : α → E
      g : α → F
      l : Filter α
      h₀ : Or (∀ (x : α), LE.le 0 (Norm.norm (f x))) (∀ (x : α), LE.le 0 (Norm.norm  …
      H : ∀ (n : Nat), Filter.Eventually (fun x => LE.le (HMul.hMul (↑n) (Norm.norm  …
      ε : Real
      ε0 : LT.lt 0 ε
      n : Nat
      hn : LT.lt (Inv.inv ε) ↑n
      hn₀ : LT.lt 0 ↑n
      ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul ε (Norm.norm  …
    -/
    refine ((isBigOWith_inv hn₀).2 (H n)).bound.mono fun x hfg => ?_
    /-
      case mpr.intro
      α : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝¹ : Norm E
      inst✝ : Norm F
      f : α → E
      g : α → F
      l : Filter α
      h₀ : Or (∀ (x : α), LE.le 0 (Norm.norm (f x))) (∀ (x : α), LE.le 0 (Norm.norm  …
      H : ∀ (n : Nat), Filter.Eventually (fun x => LE.le (HMul.hMul (↑n) (Norm.norm  …
      ε : Real
      ε0 : LT.lt 0 ε
      n : Nat
      hn : LT.lt (Inv.inv ε) ↑n
      hn₀ : LT.lt 0 ↑n
      x : α
      hfg : LE.le (Norm.norm (f x)) (HMul.hMul (Inv.inv ↑n) (Norm.norm (g x)))
      ⊢ LE.le (Norm.norm (f x)) (HMul.hMul ε (Norm.norm (g x)))
    -/
    refine hfg.trans (mul_le_mul_of_nonneg_right (inv_le_of_inv_le₀ ε0 hn.le) ?_)
    /-
      case mpr.intro
      α : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝¹ : Norm E
      inst✝ : Norm F
      f : α → E
      g : α → F
      l : Filter α
      h₀ : Or (∀ (x : α), LE.le 0 (Norm.norm (f x))) (∀ (x : α), LE.le 0 (Norm.norm  …
      H : ∀ (n : Nat), Filter.Eventually (fun x => LE.le (HMul.hMul (↑n) (Norm.norm  …
      ε : Real
      ε0 : LT.lt 0 ε
      n : Nat
      hn : LT.lt (Inv.inv ε) ↑n
      hn₀ : LT.lt 0 ↑n
      x : α
      hfg : LE.le (Norm.norm (f x)) (HMul.hMul (Inv.inv ↑n) (Norm.norm (g x)))
      ⊢ LE.le 0 (Norm.norm (g x))
    -/
    refine h₀.elim (fun hf => nonneg_of_mul_nonneg_right ((hf x).trans hfg) ?_) fun h => h x
    /-
      case mpr.intro
      α : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝¹ : Norm E
      inst✝ : Norm F
      f : α → E
      g : α → F
      l : Filter α
      h₀ : Or (∀ (x : α), LE.le 0 (Norm.norm (f x))) (∀ (x : α), LE.le 0 (Norm.norm  …
      H : ∀ (n : Nat), Filter.Eventually (fun x => LE.le (HMul.hMul (↑n) (Norm.norm  …
      ε : Real
      ε0 : LT.lt 0 ε
      n : Nat
      hn : LT.lt (Inv.inv ε) ↑n
      hn₀ : LT.lt 0 ↑n
      x : α
      hfg : LE.le (Norm.norm (f x)) (HMul.hMul (Inv.inv ↑n) (Norm.norm (g x)))
      hf : ∀ (x : α), LE.le 0 (Norm.norm (f x))
      ⊢ LT.lt 0 (Inv.inv ↑n)
    -/
    exact inv_pos.2 hn₀
    /-
      🎉 no goals
    -/


theorem isLittleO_iff_nat_mul_le : f =o[l] g' ↔ ∀ n : ℕ, ∀ᶠ x in l, ↑n * ‖f x‖ ≤ ‖g' x‖ :=
  isLittleO_iff_nat_mul_le_aux (Or.inr fun _x => norm_nonneg _)


theorem isLittleO_iff_nat_mul_le' : f' =o[l] g ↔ ∀ n : ℕ, ∀ᶠ x in l, ↑n * ‖f' x‖ ≤ ‖g x‖ :=
  isLittleO_iff_nat_mul_le_aux (Or.inl fun _x => norm_nonneg _)


@[nontriviality]
theorem isLittleO_of_subsingleton [Subsingleton E'] : f' =o[l] g' :=
                                    /-
                                      α : Type u_1
                                      E' : Type u_6
                                      F' : Type u_7
                                      inst✝² : SeminormedAddCommGroup E'
                                      inst✝¹ : SeminormedAddCommGroup F'
                                      f' : α → E'
                                      g' : α → F'
                                      l : Filter α
                                      inst✝ : Subsingleton E'
                                      c : Real
                                      hc : LT.lt 0 c
                                      ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (f' x)) (HMul.hMul c (Norm.norm …
                                    -/
  IsLittleO.of_bound fun c hc => by simp [Subsingleton.elim (f' _) 0, mul_nonneg hc.le]
                                    /-
                                      🎉 no goals
                                    -/


@[nontriviality]
theorem isBigO_of_subsingleton [Subsingleton E'] : f' =O[l] g' :=
  isLittleO_of_subsingleton.isBigO


theorem isBigOWith_congr (hc : c₁ = c₂) (hf : f₁ =ᶠ[l] f₂) (hg : g₁ =ᶠ[l] g₂) :
    IsBigOWith c₁ l f₁ g₁ ↔ IsBigOWith c₂ l f₂ g₂ := by
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    c₁ c₂ : Real
    l : Filter α
    f₁ f₂ : α → E
    g₁ g₂ : α → F
    hc : Eq c₁ c₂
    hf : l.EventuallyEq f₁ f₂
    hg : l.EventuallyEq g₁ g₂
    ⊢ Iff (Asymptotics.IsBigOWith c₁ l f₁ g₁) (Asymptotics.IsBigOWith c₂ l f₂ g₂)
  -/
  simp only [IsBigOWith_def]
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    c₁ c₂ : Real
    l : Filter α
    f₁ f₂ : α → E
    g₁ g₂ : α → F
    hc : Eq c₁ c₂
    hf : l.EventuallyEq f₁ f₂
    hg : l.EventuallyEq g₁ g₂
    ⊢ Iff (Filter.Eventually (fun x => LE.le (Norm.norm (f₁ x)) (HMul.hMul c₁ (Nor …
  -/
  subst c₂
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    c₁ : Real
    l : Filter α
    f₁ f₂ : α → E
    g₁ g₂ : α → F
    hf : l.EventuallyEq f₁ f₂
    hg : l.EventuallyEq g₁ g₂
    ⊢ Iff (Filter.Eventually (fun x => LE.le (Norm.norm (f₁ x)) (HMul.hMul c₁ (Nor …
  -/
  apply Filter.eventually_congr
  /-
    case h
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    c₁ : Real
    l : Filter α
    f₁ f₂ : α → E
    g₁ g₂ : α → F
    hf : l.EventuallyEq f₁ f₂
    hg : l.EventuallyEq g₁ g₂
    ⊢ Filter.Eventually (fun x => Iff (LE.le (Norm.norm (f₁ x)) (HMul.hMul c₁ (Nor …
  -/
  filter_upwards [hf, hg] with _ e₁ e₂
  /-
    case h
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    c₁ : Real
    l : Filter α
    f₁ f₂ : α → E
    g₁ g₂ : α → F
    hf : l.EventuallyEq f₁ f₂
    hg : l.EventuallyEq g₁ g₂
    a✝ : α
    e₁ : Eq (f₁ a✝) (f₂ a✝)
    e₂ : Eq (g₁ a✝) (g₂ a✝)
    ⊢ Iff (LE.le (Norm.norm (f₁ a✝)) (HMul.hMul c₁ (Norm.norm (g₁ a✝)))) (LE.le (N …
  -/
  rw [e₁, e₂]
  /-
    🎉 no goals
  -/


theorem IsBigOWith.congr' (h : IsBigOWith c₁ l f₁ g₁) (hc : c₁ = c₂) (hf : f₁ =ᶠ[l] f₂)
    (hg : g₁ =ᶠ[l] g₂) : IsBigOWith c₂ l f₂ g₂ :=
  (isBigOWith_congr hc hf hg).mp h


theorem IsBigOWith.congr (h : IsBigOWith c₁ l f₁ g₁) (hc : c₁ = c₂) (hf : ∀ x, f₁ x = f₂ x)
    (hg : ∀ x, g₁ x = g₂ x) : IsBigOWith c₂ l f₂ g₂ :=
  h.congr' hc (univ_mem' hf) (univ_mem' hg)


theorem IsBigOWith.congr_left (h : IsBigOWith c l f₁ g) (hf : ∀ x, f₁ x = f₂ x) :
    IsBigOWith c l f₂ g :=
  h.congr rfl hf fun _ => rfl


theorem IsBigOWith.congr_right (h : IsBigOWith c l f g₁) (hg : ∀ x, g₁ x = g₂ x) :
    IsBigOWith c l f g₂ :=
  h.congr rfl (fun _ => rfl) hg


theorem IsBigOWith.congr_const (h : IsBigOWith c₁ l f g) (hc : c₁ = c₂) : IsBigOWith c₂ l f g :=
  h.congr hc (fun _ => rfl) fun _ => rfl


theorem isBigO_congr (hf : f₁ =ᶠ[l] f₂) (hg : g₁ =ᶠ[l] g₂) : f₁ =O[l] g₁ ↔ f₂ =O[l] g₂ := by
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    l : Filter α
    f₁ f₂ : α → E
    g₁ g₂ : α → F
    hf : l.EventuallyEq f₁ f₂
    hg : l.EventuallyEq g₁ g₂
    ⊢ Iff (Asymptotics.IsBigO l f₁ g₁) (Asymptotics.IsBigO l f₂ g₂)
  -/
  simp only [IsBigO_def]
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    l : Filter α
    f₁ f₂ : α → E
    g₁ g₂ : α → F
    hf : l.EventuallyEq f₁ f₂
    hg : l.EventuallyEq g₁ g₂
    ⊢ Iff (Exists fun c => Asymptotics.IsBigOWith c l f₁ g₁) (Exists fun c => Asym …
  -/
  exact exists_congr fun c => isBigOWith_congr rfl hf hg
  /-
    🎉 no goals
  -/


theorem IsBigO.congr' (h : f₁ =O[l] g₁) (hf : f₁ =ᶠ[l] f₂) (hg : g₁ =ᶠ[l] g₂) : f₂ =O[l] g₂ :=
  (isBigO_congr hf hg).mp h


theorem IsBigO.congr (h : f₁ =O[l] g₁) (hf : ∀ x, f₁ x = f₂ x) (hg : ∀ x, g₁ x = g₂ x) :
    f₂ =O[l] g₂ :=
  h.congr' (univ_mem' hf) (univ_mem' hg)


theorem IsBigO.congr_left (h : f₁ =O[l] g) (hf : ∀ x, f₁ x = f₂ x) : f₂ =O[l] g :=
  h.congr hf fun _ => rfl


theorem IsBigO.congr_right (h : f =O[l] g₁) (hg : ∀ x, g₁ x = g₂ x) : f =O[l] g₂ :=
  h.congr (fun _ => rfl) hg


theorem isLittleO_congr (hf : f₁ =ᶠ[l] f₂) (hg : g₁ =ᶠ[l] g₂) : f₁ =o[l] g₁ ↔ f₂ =o[l] g₂ := by
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    l : Filter α
    f₁ f₂ : α → E
    g₁ g₂ : α → F
    hf : l.EventuallyEq f₁ f₂
    hg : l.EventuallyEq g₁ g₂
    ⊢ Iff (Asymptotics.IsLittleO l f₁ g₁) (Asymptotics.IsLittleO l f₂ g₂)
  -/
  simp only [IsLittleO_def]
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    l : Filter α
    f₁ f₂ : α → E
    g₁ g₂ : α → F
    hf : l.EventuallyEq f₁ f₂
    hg : l.EventuallyEq g₁ g₂
    ⊢ Iff (∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l f₁ g₁) (∀ ⦃c : Rea …
  -/
  exact forall₂_congr fun c _hc => isBigOWith_congr (Eq.refl c) hf hg
  /-
    🎉 no goals
  -/


theorem IsLittleO.congr' (h : f₁ =o[l] g₁) (hf : f₁ =ᶠ[l] f₂) (hg : g₁ =ᶠ[l] g₂) : f₂ =o[l] g₂ :=
  (isLittleO_congr hf hg).mp h


theorem IsLittleO.congr (h : f₁ =o[l] g₁) (hf : ∀ x, f₁ x = f₂ x) (hg : ∀ x, g₁ x = g₂ x) :
    f₂ =o[l] g₂ :=
  h.congr' (univ_mem' hf) (univ_mem' hg)


theorem IsLittleO.congr_left (h : f₁ =o[l] g) (hf : ∀ x, f₁ x = f₂ x) : f₂ =o[l] g :=
  h.congr hf fun _ => rfl


theorem IsLittleO.congr_right (h : f =o[l] g₁) (hg : ∀ x, g₁ x = g₂ x) : f =o[l] g₂ :=
  h.congr (fun _ => rfl) hg


@[trans]
theorem _root_.Filter.EventuallyEq.trans_isBigO {f₁ f₂ : α → E} {g : α → F} (hf : f₁ =ᶠ[l] f₂)
    (h : f₂ =O[l] g) : f₁ =O[l] g :=
  h.congr' hf.symm EventuallyEq.rfl


instance transEventuallyEqIsBigO :
    @Trans (α → E) (α → E) (α → F) (· =ᶠ[l] ·) (· =O[l] ·) (· =O[l] ·) where
  trans := Filter.EventuallyEq.trans_isBigO


@[trans]
theorem _root_.Filter.EventuallyEq.trans_isLittleO {f₁ f₂ : α → E} {g : α → F} (hf : f₁ =ᶠ[l] f₂)
    (h : f₂ =o[l] g) : f₁ =o[l] g :=
  h.congr' hf.symm EventuallyEq.rfl


instance transEventuallyEqIsLittleO :
    @Trans (α → E) (α → E) (α → F) (· =ᶠ[l] ·) (· =o[l] ·) (· =o[l] ·) where
  trans := Filter.EventuallyEq.trans_isLittleO


@[trans]
theorem IsBigO.trans_eventuallyEq {f : α → E} {g₁ g₂ : α → F} (h : f =O[l] g₁) (hg : g₁ =ᶠ[l] g₂) :
    f =O[l] g₂ :=
  h.congr' EventuallyEq.rfl hg


instance transIsBigOEventuallyEq :
    @Trans (α → E) (α → F) (α → F) (· =O[l] ·) (· =ᶠ[l] ·) (· =O[l] ·) where
  trans := IsBigO.trans_eventuallyEq


@[trans]
theorem IsLittleO.trans_eventuallyEq {f : α → E} {g₁ g₂ : α → F} (h : f =o[l] g₁)
    (hg : g₁ =ᶠ[l] g₂) : f =o[l] g₂ :=
  h.congr' EventuallyEq.rfl hg


instance transIsLittleOEventuallyEq :
    @Trans (α → E) (α → F) (α → F) (· =o[l] ·) (· =ᶠ[l] ·) (· =o[l] ·) where
  trans := IsLittleO.trans_eventuallyEq


theorem IsBigOWith.comp_tendsto (hcfg : IsBigOWith c l f g) {k : β → α} {l' : Filter β}
    (hk : Tendsto k l' l) : IsBigOWith c l' (f ∘ k) (g ∘ k) :=
  IsBigOWith.of_bound <| hk hcfg.bound


theorem IsBigO.comp_tendsto (hfg : f =O[l] g) {k : β → α} {l' : Filter β} (hk : Tendsto k l' l) :
    (f ∘ k) =O[l'] (g ∘ k) :=
  isBigO_iff_isBigOWith.2 <| hfg.isBigOWith.imp fun _c h => h.comp_tendsto hk


theorem IsLittleO.comp_tendsto (hfg : f =o[l] g) {k : β → α} {l' : Filter β} (hk : Tendsto k l' l) :
    (f ∘ k) =o[l'] (g ∘ k) :=
  IsLittleO.of_isBigOWith fun _c cpos => (hfg.forall_isBigOWith cpos).comp_tendsto hk


@[simp]
theorem isBigOWith_map {k : β → α} {l : Filter β} :
    IsBigOWith c (map k l) f g ↔ IsBigOWith c l (f ∘ k) (g ∘ k) := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    c : Real
    f : α → E
    g : α → F
    k : β → α
    l : Filter β
    ⊢ Iff (Asymptotics.IsBigOWith c (Filter.map k l) f g) (Asymptotics.IsBigOWith  …
  -/
  simp only [IsBigOWith_def]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    c : Real
    f : α → E
    g : α → F
    k : β → α
    l : Filter β
    ⊢ Iff (Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul c (Norm. …
  -/
  exact eventually_map
  /-
    🎉 no goals
  -/


@[simp]
theorem isBigO_map {k : β → α} {l : Filter β} : f =O[map k l] g ↔ (f ∘ k) =O[l] (g ∘ k) := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    k : β → α
    l : Filter β
    ⊢ Iff (Asymptotics.IsBigO (Filter.map k l) f g) (Asymptotics.IsBigO l (Functio …
  -/
  simp only [IsBigO_def, isBigOWith_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem isLittleO_map {k : β → α} {l : Filter β} : f =o[map k l] g ↔ (f ∘ k) =o[l] (g ∘ k) := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    k : β → α
    l : Filter β
    ⊢ Iff (Asymptotics.IsLittleO (Filter.map k l) f g) (Asymptotics.IsLittleO l (F …
  -/
  simp only [IsLittleO_def, isBigOWith_map]
  /-
    🎉 no goals
  -/


theorem IsBigOWith.mono (h : IsBigOWith c l' f g) (hl : l ≤ l') : IsBigOWith c l f g :=
  IsBigOWith.of_bound <| hl h.bound


theorem IsBigO.mono (h : f =O[l'] g) (hl : l ≤ l') : f =O[l] g :=
  isBigO_iff_isBigOWith.2 <| h.isBigOWith.imp fun _c h => h.mono hl


theorem IsLittleO.mono (h : f =o[l'] g) (hl : l ≤ l') : f =o[l] g :=
  IsLittleO.of_isBigOWith fun _c cpos => (h.forall_isBigOWith cpos).mono hl


theorem IsBigOWith.trans (hfg : IsBigOWith c l f g) (hgk : IsBigOWith c' l g k) (hc : 0 ≤ c) :
    IsBigOWith (c * c') l f k := by
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝² : Norm E
    inst✝¹ : Norm F
    inst✝ : Norm G
    c c' : Real
    f : α → E
    g : α → F
    k : α → G
    l : Filter α
    hfg : Asymptotics.IsBigOWith c l f g
    hgk : Asymptotics.IsBigOWith c' l g k
    hc : LE.le 0 c
    ⊢ Asymptotics.IsBigOWith (HMul.hMul c c') l f k
  -/
  simp only [IsBigOWith_def] at *
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝² : Norm E
    inst✝¹ : Norm F
    inst✝ : Norm G
    c c' : Real
    f : α → E
    g : α → F
    k : α → G
    l : Filter α
    hc : LE.le 0 c
    hfg : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul c (Norm.n …
    hgk : Filter.Eventually (fun x => LE.le (Norm.norm (g x)) (HMul.hMul c' (Norm. …
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.hMul (HMul.hMul c  …
  -/
  filter_upwards [hfg, hgk] with x hx hx'
  calc
    ‖f x‖ ≤ c * ‖g x‖ := hx
    _ ≤ c * (c' * ‖k x‖) := by gcongr
    _ = c * c' * ‖k x‖ := (mul_assoc _ _ _).symm


@[trans]
theorem IsBigO.trans {f : α → E} {g : α → F'} {k : α → G} (hfg : f =O[l] g) (hgk : g =O[l] k) :
    f =O[l] k :=
  let ⟨_c, cnonneg, hc⟩ := hfg.exists_nonneg
  let ⟨_c', hc'⟩ := hgk.isBigOWith
  (hc.trans hc' cnonneg).isBigO


instance transIsBigOIsBigO :
    @Trans (α → E) (α → F') (α → G) (· =O[l] ·) (· =O[l] ·) (· =O[l] ·) where
  trans := IsBigO.trans


theorem IsLittleO.trans_isBigOWith (hfg : f =o[l] g) (hgk : IsBigOWith c l g k) (hc : 0 < c) :
    f =o[l] k := by
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝² : Norm E
    inst✝¹ : Norm F
    inst✝ : Norm G
    c : Real
    f : α → E
    g : α → F
    k : α → G
    l : Filter α
    hfg : Asymptotics.IsLittleO l f g
    hgk : Asymptotics.IsBigOWith c l g k
    hc : LT.lt 0 c
    ⊢ Asymptotics.IsLittleO l f k
  -/
  simp only [IsLittleO_def] at *
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝² : Norm E
    inst✝¹ : Norm F
    inst✝ : Norm G
    c : Real
    f : α → E
    g : α → F
    k : α → G
    l : Filter α
    hgk : Asymptotics.IsBigOWith c l g k
    hc : LT.lt 0 c
    hfg : ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l f g
    ⊢ ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l f k
  -/
  intro c' c'pos
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝² : Norm E
    inst✝¹ : Norm F
    inst✝ : Norm G
    c : Real
    f : α → E
    g : α → F
    k : α → G
    l : Filter α
    hgk : Asymptotics.IsBigOWith c l g k
    hc : LT.lt 0 c
    hfg : ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l f g
    c' : Real
    c'pos : LT.lt 0 c'
    ⊢ Asymptotics.IsBigOWith c' l f k
  -/
  have : 0 < c' / c := div_pos c'pos hc
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝² : Norm E
    inst✝¹ : Norm F
    inst✝ : Norm G
    c : Real
    f : α → E
    g : α → F
    k : α → G
    l : Filter α
    hgk : Asymptotics.IsBigOWith c l g k
    hc : LT.lt 0 c
    hfg : ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l f g
    c' : Real
    c'pos : LT.lt 0 c'
    this : LT.lt 0 (HDiv.hDiv c' c)
    ⊢ Asymptotics.IsBigOWith c' l f k
  -/
  exact ((hfg this).trans hgk this.le).congr_const (div_mul_cancel₀ _ hc.ne')
  /-
    🎉 no goals
  -/


@[trans]
theorem IsLittleO.trans_isBigO {f : α → E} {g : α → F} {k : α → G'} (hfg : f =o[l] g)
    (hgk : g =O[l] k) : f =o[l] k :=
  let ⟨_c, cpos, hc⟩ := hgk.exists_pos
  hfg.trans_isBigOWith hc cpos


instance transIsLittleOIsBigO :
    @Trans (α → E) (α → F) (α → G') (· =o[l] ·) (· =O[l] ·) (· =o[l] ·) where
  trans := IsLittleO.trans_isBigO


theorem IsBigOWith.trans_isLittleO (hfg : IsBigOWith c l f g) (hgk : g =o[l] k) (hc : 0 < c) :
    f =o[l] k := by
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝² : Norm E
    inst✝¹ : Norm F
    inst✝ : Norm G
    c : Real
    f : α → E
    g : α → F
    k : α → G
    l : Filter α
    hfg : Asymptotics.IsBigOWith c l f g
    hgk : Asymptotics.IsLittleO l g k
    hc : LT.lt 0 c
    ⊢ Asymptotics.IsLittleO l f k
  -/
  simp only [IsLittleO_def] at *
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝² : Norm E
    inst✝¹ : Norm F
    inst✝ : Norm G
    c : Real
    f : α → E
    g : α → F
    k : α → G
    l : Filter α
    hfg : Asymptotics.IsBigOWith c l f g
    hc : LT.lt 0 c
    hgk : ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l g k
    ⊢ ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l f k
  -/
  intro c' c'pos
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝² : Norm E
    inst✝¹ : Norm F
    inst✝ : Norm G
    c : Real
    f : α → E
    g : α → F
    k : α → G
    l : Filter α
    hfg : Asymptotics.IsBigOWith c l f g
    hc : LT.lt 0 c
    hgk : ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l g k
    c' : Real
    c'pos : LT.lt 0 c'
    ⊢ Asymptotics.IsBigOWith c' l f k
  -/
  have : 0 < c' / c := div_pos c'pos hc
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝² : Norm E
    inst✝¹ : Norm F
    inst✝ : Norm G
    c : Real
    f : α → E
    g : α → F
    k : α → G
    l : Filter α
    hfg : Asymptotics.IsBigOWith c l f g
    hc : LT.lt 0 c
    hgk : ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l g k
    c' : Real
    c'pos : LT.lt 0 c'
    this : LT.lt 0 (HDiv.hDiv c' c)
    ⊢ Asymptotics.IsBigOWith c' l f k
  -/
  exact (hfg.trans (hgk this) hc.le).congr_const (mul_div_cancel₀ _ hc.ne')
  /-
    🎉 no goals
  -/


@[trans]
theorem IsBigO.trans_isLittleO {f : α → E} {g : α → F'} {k : α → G} (hfg : f =O[l] g)
    (hgk : g =o[l] k) : f =o[l] k :=
  let ⟨_c, cpos, hc⟩ := hfg.exists_pos
  hc.trans_isLittleO hgk cpos


instance transIsBigOIsLittleO :
    @Trans (α → E) (α → F') (α → G) (· =O[l] ·) (· =o[l] ·) (· =o[l] ·) where
  trans := IsBigO.trans_isLittleO


@[trans]
theorem IsLittleO.trans {f : α → E} {g : α → F} {k : α → G} (hfg : f =o[l] g) (hgk : g =o[l] k) :
    f =o[l] k :=
  hfg.trans_isBigOWith hgk.isBigOWith one_pos


instance transIsLittleOIsLittleO :
    @Trans (α → E) (α → F) (α → G) (· =o[l] ·) (· =o[l] ·) (· =o[l] ·) where
  trans := IsLittleO.trans


theorem _root_.Filter.Eventually.trans_isBigO {f : α → E} {g : α → F'} {k : α → G}
    (hfg : ∀ᶠ x in l, ‖f x‖ ≤ ‖g x‖) (hgk : g =O[l] k) : f =O[l] k :=
  (IsBigO.of_bound' hfg).trans hgk


theorem _root_.Filter.Eventually.isBigO {f : α → E} {g : α → ℝ} {l : Filter α}
    (hfg : ∀ᶠ x in l, ‖f x‖ ≤ g x) : f =O[l] g :=
  IsBigO.of_bound' <| hfg.mono fun _x hx => hx.trans <| Real.le_norm_self _


theorem isBigOWith_of_le' (hfg : ∀ x, ‖f x‖ ≤ c * ‖g x‖) : IsBigOWith c l f g :=
  IsBigOWith.of_bound <| univ_mem' hfg


theorem isBigOWith_of_le (hfg : ∀ x, ‖f x‖ ≤ ‖g x‖) : IsBigOWith 1 l f g :=
  isBigOWith_of_le' l fun x => by
    /-
      α : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝¹ : Norm E
      inst✝ : Norm F
      f : α → E
      g : α → F
      l : Filter α
      hfg : ∀ (x : α), LE.le (Norm.norm (f x)) (Norm.norm (g x))
      x : α
      ⊢ LE.le (Norm.norm (f x)) (HMul.hMul 1 (Norm.norm (g x)))
    -/
    rw [one_mul]
    /-
      α : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝¹ : Norm E
      inst✝ : Norm F
      f : α → E
      g : α → F
      l : Filter α
      hfg : ∀ (x : α), LE.le (Norm.norm (f x)) (Norm.norm (g x))
      x : α
      ⊢ LE.le (Norm.norm (f x)) (Norm.norm (g x))
    -/
    exact hfg x
    /-
      🎉 no goals
    -/


theorem isBigO_of_le' (hfg : ∀ x, ‖f x‖ ≤ c * ‖g x‖) : f =O[l] g :=
  (isBigOWith_of_le' l hfg).isBigO


theorem isBigO_of_le (hfg : ∀ x, ‖f x‖ ≤ ‖g x‖) : f =O[l] g :=
  (isBigOWith_of_le l hfg).isBigO


theorem isBigOWith_refl (f : α → E) (l : Filter α) : IsBigOWith 1 l f f :=
  isBigOWith_of_le l fun _ => le_rfl


theorem isBigO_refl (f : α → E) (l : Filter α) : f =O[l] f :=
  (isBigOWith_refl f l).isBigO


theorem _root_.Filter.EventuallyEq.isBigO {f₁ f₂ : α → E} (hf : f₁ =ᶠ[l] f₂) : f₁ =O[l] f₂ :=
  hf.trans_isBigO (isBigO_refl _ _)


theorem IsBigOWith.trans_le (hfg : IsBigOWith c l f g) (hgk : ∀ x, ‖g x‖ ≤ ‖k x‖) (hc : 0 ≤ c) :
    IsBigOWith c l f k :=
  (hfg.trans (isBigOWith_of_le l hgk) hc).congr_const <| mul_one c


theorem IsBigO.trans_le (hfg : f =O[l] g') (hgk : ∀ x, ‖g' x‖ ≤ ‖k x‖) : f =O[l] k :=
  hfg.trans (isBigO_of_le l hgk)


theorem IsLittleO.trans_le (hfg : f =o[l] g) (hgk : ∀ x, ‖g x‖ ≤ ‖k x‖) : f =o[l] k :=
  hfg.trans_isBigOWith (isBigOWith_of_le _ hgk) zero_lt_one


theorem isLittleO_irrefl' (h : ∃ᶠ x in l, ‖f' x‖ ≠ 0) : ¬f' =o[l] f' := by
  /-
    α : Type u_1
    E' : Type u_6
    inst✝ : SeminormedAddCommGroup E'
    f' : α → E'
    l : Filter α
    h : Filter.Frequently (fun x => Ne (Norm.norm (f' x)) 0) l
    ⊢ Not (Asymptotics.IsLittleO l f' f')
  -/
  intro ho
  /-
    α : Type u_1
    E' : Type u_6
    inst✝ : SeminormedAddCommGroup E'
    f' : α → E'
    l : Filter α
    h : Filter.Frequently (fun x => Ne (Norm.norm (f' x)) 0) l
    ho : Asymptotics.IsLittleO l f' f'
    ⊢ False
  -/
  rcases ((ho.bound one_half_pos).and_frequently h).exists with ⟨x, hle, hne⟩
  /-
    case intro.intro
    α : Type u_1
    E' : Type u_6
    inst✝ : SeminormedAddCommGroup E'
    f' : α → E'
    l : Filter α
    h : Filter.Frequently (fun x => Ne (Norm.norm (f' x)) 0) l
    ho : Asymptotics.IsLittleO l f' f'
    x : α
    hle : LE.le (Norm.norm (f' x)) (HMul.hMul (1 / 2) (Norm.norm (f' x)))
    hne : Ne (Norm.norm (f' x)) 0
    ⊢ False
  -/
  rw [one_div, ← div_eq_inv_mul] at hle
  /-
    case intro.intro
    α : Type u_1
    E' : Type u_6
    inst✝ : SeminormedAddCommGroup E'
    f' : α → E'
    l : Filter α
    h : Filter.Frequently (fun x => Ne (Norm.norm (f' x)) 0) l
    ho : Asymptotics.IsLittleO l f' f'
    x : α
    hle : LE.le (Norm.norm (f' x)) (HDiv.hDiv (Norm.norm (f' x)) 2)
    hne : Ne (Norm.norm (f' x)) 0
    ⊢ False
  -/
  exact (half_lt_self (lt_of_le_of_ne (norm_nonneg _) hne.symm)).not_le hle
  /-
    🎉 no goals
  -/


theorem isLittleO_irrefl (h : ∃ᶠ x in l, f'' x ≠ 0) : ¬f'' =o[l] f'' :=
  isLittleO_irrefl' <| h.mono fun _x => norm_ne_zero_iff.mpr


theorem IsBigO.not_isLittleO (h : f'' =O[l] g') (hf : ∃ᶠ x in l, f'' x ≠ 0) :
    ¬g' =o[l] f'' := fun h' =>
  isLittleO_irrefl hf (h.trans_isLittleO h')


theorem IsLittleO.not_isBigO (h : f'' =o[l] g') (hf : ∃ᶠ x in l, f'' x ≠ 0) :
    ¬g' =O[l] f'' := fun h' =>
  isLittleO_irrefl hf (h.trans_isBigO h')


@[simp]
theorem isBigOWith_bot : IsBigOWith c ⊥ f g :=
  IsBigOWith.of_bound <| trivial


@[simp]
theorem isBigO_bot : f =O[⊥] g :=
  (isBigOWith_bot 1 f g).isBigO


@[simp]
theorem isLittleO_bot : f =o[⊥] g :=
  IsLittleO.of_isBigOWith fun c _ => isBigOWith_bot c f g


@[simp]
theorem isBigOWith_pure {x} : IsBigOWith c (pure x) f g ↔ ‖f x‖ ≤ c * ‖g x‖ :=
  isBigOWith_iff


theorem IsBigOWith.sup (h : IsBigOWith c l f g) (h' : IsBigOWith c l' f g) :
    IsBigOWith c (l ⊔ l') f g :=
  IsBigOWith.of_bound <| mem_sup.2 ⟨h.bound, h'.bound⟩


theorem IsBigOWith.sup' (h : IsBigOWith c l f g') (h' : IsBigOWith c' l' f g') :
    IsBigOWith (max c c') (l ⊔ l') f g' :=
  IsBigOWith.of_bound <|
    mem_sup.2 ⟨(h.weaken <| le_max_left c c').bound, (h'.weaken <| le_max_right c c').bound⟩


theorem IsBigO.sup (h : f =O[l] g') (h' : f =O[l'] g') : f =O[l ⊔ l'] g' :=
  let ⟨_c, hc⟩ := h.isBigOWith
  let ⟨_c', hc'⟩ := h'.isBigOWith
  (hc.sup' hc').isBigO


theorem IsLittleO.sup (h : f =o[l] g) (h' : f =o[l'] g) : f =o[l ⊔ l'] g :=
  IsLittleO.of_isBigOWith fun _c cpos => (h.forall_isBigOWith cpos).sup (h'.forall_isBigOWith cpos)


@[simp]
theorem isBigO_sup : f =O[l ⊔ l'] g' ↔ f =O[l] g' ∧ f =O[l'] g' :=
  ⟨fun h => ⟨h.mono le_sup_left, h.mono le_sup_right⟩, fun h => h.1.sup h.2⟩


@[simp]
theorem isLittleO_sup : f =o[l ⊔ l'] g ↔ f =o[l] g ∧ f =o[l'] g :=
  ⟨fun h => ⟨h.mono le_sup_left, h.mono le_sup_right⟩, fun h => h.1.sup h.2⟩


theorem isBigOWith_insert [TopologicalSpace α] {x : α} {s : Set α} {C : ℝ} {g : α → E} {g' : α → F}
    (h : ‖g x‖ ≤ C * ‖g' x‖) : IsBigOWith C (𝓝[insert x s] x) g g' ↔
    IsBigOWith C (𝓝[s] x) g g' := by
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝² : Norm E
    inst✝¹ : Norm F
    inst✝ : TopologicalSpace α
    x : α
    s : Set α
    C : Real
    g : α → E
    g' : α → F
    h : LE.le (Norm.norm (g x)) (HMul.hMul C (Norm.norm (g' x)))
    ⊢ Iff (Asymptotics.IsBigOWith C (nhdsWithin x (Insert.insert x s)) g g') (Asym …
  -/
  simp_rw [IsBigOWith_def, nhdsWithin_insert, eventually_sup, eventually_pure, h, true_and]
  /-
    🎉 no goals
  -/


protected theorem IsBigOWith.insert [TopologicalSpace α] {x : α} {s : Set α} {C : ℝ} {g : α → E}
    {g' : α → F} (h1 : IsBigOWith C (𝓝[s] x) g g') (h2 : ‖g x‖ ≤ C * ‖g' x‖) :
    IsBigOWith C (𝓝[insert x s] x) g g' :=
  (isBigOWith_insert h2).mpr h1


theorem isLittleO_insert [TopologicalSpace α] {x : α} {s : Set α} {g : α → E'} {g' : α → F'}
    (h : g x = 0) : g =o[𝓝[insert x s] x] g' ↔ g =o[𝓝[s] x] g' := by
  /-
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    inst✝² : SeminormedAddCommGroup E'
    inst✝¹ : SeminormedAddCommGroup F'
    inst✝ : TopologicalSpace α
    x : α
    s : Set α
    g : α → E'
    g' : α → F'
    h : Eq (g x) 0
    ⊢ Iff (Asymptotics.IsLittleO (nhdsWithin x (Insert.insert x s)) g g') (Asympto …
  -/
  simp_rw [IsLittleO_def]
  /-
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    inst✝² : SeminormedAddCommGroup E'
    inst✝¹ : SeminormedAddCommGroup F'
    inst✝ : TopologicalSpace α
    x : α
    s : Set α
    g : α → E'
    g' : α → F'
    h : Eq (g x) 0
    ⊢ Iff (∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c (nhdsWithin x (Inser …
  -/
  refine forall_congr' fun c => forall_congr' fun hc => ?_
  /-
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    inst✝² : SeminormedAddCommGroup E'
    inst✝¹ : SeminormedAddCommGroup F'
    inst✝ : TopologicalSpace α
    x : α
    s : Set α
    g : α → E'
    g' : α → F'
    h : Eq (g x) 0
    c : Real
    hc : LT.lt 0 c
    ⊢ Iff (Asymptotics.IsBigOWith c (nhdsWithin x (Insert.insert x s)) g g') (Asym …
  -/
  rw [isBigOWith_insert]
  /-
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    inst✝² : SeminormedAddCommGroup E'
    inst✝¹ : SeminormedAddCommGroup F'
    inst✝ : TopologicalSpace α
    x : α
    s : Set α
    g : α → E'
    g' : α → F'
    h : Eq (g x) 0
    c : Real
    hc : LT.lt 0 c
    ⊢ LE.le (Norm.norm (g x)) (HMul.hMul c (Norm.norm (g' x)))
  -/
  rw [h, norm_zero]
  /-
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    inst✝² : SeminormedAddCommGroup E'
    inst✝¹ : SeminormedAddCommGroup F'
    inst✝ : TopologicalSpace α
    x : α
    s : Set α
    g : α → E'
    g' : α → F'
    h : Eq (g x) 0
    c : Real
    hc : LT.lt 0 c
    ⊢ LE.le 0 (HMul.hMul c (Norm.norm (g' x)))
  -/
  exact mul_nonneg hc.le (norm_nonneg _)
  /-
    🎉 no goals
  -/


protected theorem IsLittleO.insert [TopologicalSpace α] {x : α} {s : Set α} {g : α → E'}
    {g' : α → F'} (h1 : g =o[𝓝[s] x] g') (h2 : g x = 0) : g =o[𝓝[insert x s] x] g' :=
  (isLittleO_insert h2).mpr h1


@[simp]
theorem isBigOWith_norm_right : (IsBigOWith c l f fun x => ‖g' x‖) ↔ IsBigOWith c l f g' := by
  /-
    α : Type u_1
    E : Type u_3
    F' : Type u_7
    inst✝¹ : Norm E
    inst✝ : SeminormedAddCommGroup F'
    c : Real
    f : α → E
    g' : α → F'
    l : Filter α
    ⊢ Iff (Asymptotics.IsBigOWith c l f fun x => Norm.norm (g' x)) (Asymptotics.Is …
  -/
  simp only [IsBigOWith_def, norm_norm]
  /-
    🎉 no goals
  -/


@[simp]
theorem isBigOWith_abs_right : (IsBigOWith c l f fun x => |u x|) ↔ IsBigOWith c l f u :=
  @isBigOWith_norm_right _ _ _ _ _ _ f u l


alias ⟨IsBigOWith.of_norm_right, IsBigOWith.norm_right⟩ := isBigOWith_norm_right


alias ⟨IsBigOWith.of_abs_right, IsBigOWith.abs_right⟩ := isBigOWith_abs_right


@[simp]
theorem isBigO_norm_right : (f =O[l] fun x => ‖g' x‖) ↔ f =O[l] g' := by
  /-
    α : Type u_1
    E : Type u_3
    F' : Type u_7
    inst✝¹ : Norm E
    inst✝ : SeminormedAddCommGroup F'
    f : α → E
    g' : α → F'
    l : Filter α
    ⊢ Iff (Asymptotics.IsBigO l f fun x => Norm.norm (g' x)) (Asymptotics.IsBigO l …
  -/
  simp only [IsBigO_def]
  /-
    α : Type u_1
    E : Type u_3
    F' : Type u_7
    inst✝¹ : Norm E
    inst✝ : SeminormedAddCommGroup F'
    f : α → E
    g' : α → F'
    l : Filter α
    ⊢ Iff (Exists fun c => Asymptotics.IsBigOWith c l f fun x => Norm.norm (g' x)) …
  -/
  exact exists_congr fun _ => isBigOWith_norm_right
  /-
    🎉 no goals
  -/


@[simp]
theorem isBigO_abs_right : (f =O[l] fun x => |u x|) ↔ f =O[l] u :=
  @isBigO_norm_right _ _ ℝ _ _ _ _ _


alias ⟨IsBigO.of_norm_right, IsBigO.norm_right⟩ := isBigO_norm_right


alias ⟨IsBigO.of_abs_right, IsBigO.abs_right⟩ := isBigO_abs_right


@[simp]
theorem isLittleO_norm_right : (f =o[l] fun x => ‖g' x‖) ↔ f =o[l] g' := by
  /-
    α : Type u_1
    E : Type u_3
    F' : Type u_7
    inst✝¹ : Norm E
    inst✝ : SeminormedAddCommGroup F'
    f : α → E
    g' : α → F'
    l : Filter α
    ⊢ Iff (Asymptotics.IsLittleO l f fun x => Norm.norm (g' x)) (Asymptotics.IsLit …
  -/
  simp only [IsLittleO_def]
  /-
    α : Type u_1
    E : Type u_3
    F' : Type u_7
    inst✝¹ : Norm E
    inst✝ : SeminormedAddCommGroup F'
    f : α → E
    g' : α → F'
    l : Filter α
    ⊢ Iff (∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l f fun x => Norm.no …
  -/
  exact forall₂_congr fun _ _ => isBigOWith_norm_right
  /-
    🎉 no goals
  -/


@[simp]
theorem isLittleO_abs_right : (f =o[l] fun x => |u x|) ↔ f =o[l] u :=
  @isLittleO_norm_right _ _ ℝ _ _ _ _ _


alias ⟨IsLittleO.of_norm_right, IsLittleO.norm_right⟩ := isLittleO_norm_right


alias ⟨IsLittleO.of_abs_right, IsLittleO.abs_right⟩ := isLittleO_abs_right


@[simp]
theorem isBigOWith_norm_left : IsBigOWith c l (fun x => ‖f' x‖) g ↔ IsBigOWith c l f' g := by
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    inst✝¹ : Norm F
    inst✝ : SeminormedAddCommGroup E'
    c : Real
    g : α → F
    f' : α → E'
    l : Filter α
    ⊢ Iff (Asymptotics.IsBigOWith c l (fun x => Norm.norm (f' x)) g) (Asymptotics. …
  -/
  simp only [IsBigOWith_def, norm_norm]
  /-
    🎉 no goals
  -/


@[simp]
theorem isBigOWith_abs_left : IsBigOWith c l (fun x => |u x|) g ↔ IsBigOWith c l u g :=
  @isBigOWith_norm_left _ _ _ _ _ _ g u l


alias ⟨IsBigOWith.of_norm_left, IsBigOWith.norm_left⟩ := isBigOWith_norm_left


alias ⟨IsBigOWith.of_abs_left, IsBigOWith.abs_left⟩ := isBigOWith_abs_left


@[simp]
theorem isBigO_norm_left : (fun x => ‖f' x‖) =O[l] g ↔ f' =O[l] g := by
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    inst✝¹ : Norm F
    inst✝ : SeminormedAddCommGroup E'
    g : α → F
    f' : α → E'
    l : Filter α
    ⊢ Iff (Asymptotics.IsBigO l (fun x => Norm.norm (f' x)) g) (Asymptotics.IsBigO …
  -/
  simp only [IsBigO_def]
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    inst✝¹ : Norm F
    inst✝ : SeminormedAddCommGroup E'
    g : α → F
    f' : α → E'
    l : Filter α
    ⊢ Iff (Exists fun c => Asymptotics.IsBigOWith c l (fun x => Norm.norm (f' x))  …
  -/
  exact exists_congr fun _ => isBigOWith_norm_left
  /-
    🎉 no goals
  -/


@[simp]
theorem isBigO_abs_left : (fun x => |u x|) =O[l] g ↔ u =O[l] g :=
  @isBigO_norm_left _ _ _ _ _ g u l


alias ⟨IsBigO.of_norm_left, IsBigO.norm_left⟩ := isBigO_norm_left


alias ⟨IsBigO.of_abs_left, IsBigO.abs_left⟩ := isBigO_abs_left


@[simp]
theorem isLittleO_norm_left : (fun x => ‖f' x‖) =o[l] g ↔ f' =o[l] g := by
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    inst✝¹ : Norm F
    inst✝ : SeminormedAddCommGroup E'
    g : α → F
    f' : α → E'
    l : Filter α
    ⊢ Iff (Asymptotics.IsLittleO l (fun x => Norm.norm (f' x)) g) (Asymptotics.IsL …
  -/
  simp only [IsLittleO_def]
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    inst✝¹ : Norm F
    inst✝ : SeminormedAddCommGroup E'
    g : α → F
    f' : α → E'
    l : Filter α
    ⊢ Iff (∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l (fun x => Norm.nor …
  -/
  exact forall₂_congr fun _ _ => isBigOWith_norm_left
  /-
    🎉 no goals
  -/


@[simp]
theorem isLittleO_abs_left : (fun x => |u x|) =o[l] g ↔ u =o[l] g :=
  @isLittleO_norm_left _ _ _ _ _ g u l


alias ⟨IsLittleO.of_norm_left, IsLittleO.norm_left⟩ := isLittleO_norm_left


alias ⟨IsLittleO.of_abs_left, IsLittleO.abs_left⟩ := isLittleO_abs_left


theorem isBigOWith_norm_norm :
    (IsBigOWith c l (fun x => ‖f' x‖) fun x => ‖g' x‖) ↔ IsBigOWith c l f' g' :=
  isBigOWith_norm_left.trans isBigOWith_norm_right


theorem isBigOWith_abs_abs :
    (IsBigOWith c l (fun x => |u x|) fun x => |v x|) ↔ IsBigOWith c l u v :=
  isBigOWith_abs_left.trans isBigOWith_abs_right


alias ⟨IsBigOWith.of_norm_norm, IsBigOWith.norm_norm⟩ := isBigOWith_norm_norm


alias ⟨IsBigOWith.of_abs_abs, IsBigOWith.abs_abs⟩ := isBigOWith_abs_abs


theorem isBigO_norm_norm : ((fun x => ‖f' x‖) =O[l] fun x => ‖g' x‖) ↔ f' =O[l] g' :=
  isBigO_norm_left.trans isBigO_norm_right


theorem isBigO_abs_abs : ((fun x => |u x|) =O[l] fun x => |v x|) ↔ u =O[l] v :=
  isBigO_abs_left.trans isBigO_abs_right


alias ⟨IsBigO.of_norm_norm, IsBigO.norm_norm⟩ := isBigO_norm_norm


alias ⟨IsBigO.of_abs_abs, IsBigO.abs_abs⟩ := isBigO_abs_abs


theorem isLittleO_norm_norm : ((fun x => ‖f' x‖) =o[l] fun x => ‖g' x‖) ↔ f' =o[l] g' :=
  isLittleO_norm_left.trans isLittleO_norm_right


theorem isLittleO_abs_abs : ((fun x => |u x|) =o[l] fun x => |v x|) ↔ u =o[l] v :=
  isLittleO_abs_left.trans isLittleO_abs_right


alias ⟨IsLittleO.of_norm_norm, IsLittleO.norm_norm⟩ := isLittleO_norm_norm


alias ⟨IsLittleO.of_abs_abs, IsLittleO.abs_abs⟩ := isLittleO_abs_abs


@[simp]
theorem isBigOWith_neg_right : (IsBigOWith c l f fun x => -g' x) ↔ IsBigOWith c l f g' := by
  /-
    α : Type u_1
    E : Type u_3
    F' : Type u_7
    inst✝¹ : Norm E
    inst✝ : SeminormedAddCommGroup F'
    c : Real
    f : α → E
    g' : α → F'
    l : Filter α
    ⊢ Iff (Asymptotics.IsBigOWith c l f fun x => Neg.neg (g' x)) (Asymptotics.IsBi …
  -/
  simp only [IsBigOWith_def, norm_neg]
  /-
    🎉 no goals
  -/


alias ⟨IsBigOWith.of_neg_right, IsBigOWith.neg_right⟩ := isBigOWith_neg_right


@[simp]
theorem isBigO_neg_right : (f =O[l] fun x => -g' x) ↔ f =O[l] g' := by
  /-
    α : Type u_1
    E : Type u_3
    F' : Type u_7
    inst✝¹ : Norm E
    inst✝ : SeminormedAddCommGroup F'
    f : α → E
    g' : α → F'
    l : Filter α
    ⊢ Iff (Asymptotics.IsBigO l f fun x => Neg.neg (g' x)) (Asymptotics.IsBigO l f …
  -/
  simp only [IsBigO_def]
  /-
    α : Type u_1
    E : Type u_3
    F' : Type u_7
    inst✝¹ : Norm E
    inst✝ : SeminormedAddCommGroup F'
    f : α → E
    g' : α → F'
    l : Filter α
    ⊢ Iff (Exists fun c => Asymptotics.IsBigOWith c l f fun x => Neg.neg (g' x)) ( …
  -/
  exact exists_congr fun _ => isBigOWith_neg_right
  /-
    🎉 no goals
  -/


alias ⟨IsBigO.of_neg_right, IsBigO.neg_right⟩ := isBigO_neg_right


@[simp]
theorem isLittleO_neg_right : (f =o[l] fun x => -g' x) ↔ f =o[l] g' := by
  /-
    α : Type u_1
    E : Type u_3
    F' : Type u_7
    inst✝¹ : Norm E
    inst✝ : SeminormedAddCommGroup F'
    f : α → E
    g' : α → F'
    l : Filter α
    ⊢ Iff (Asymptotics.IsLittleO l f fun x => Neg.neg (g' x)) (Asymptotics.IsLittl …
  -/
  simp only [IsLittleO_def]
  /-
    α : Type u_1
    E : Type u_3
    F' : Type u_7
    inst✝¹ : Norm E
    inst✝ : SeminormedAddCommGroup F'
    f : α → E
    g' : α → F'
    l : Filter α
    ⊢ Iff (∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l f fun x => Neg.neg …
  -/
  exact forall₂_congr fun _ _ => isBigOWith_neg_right
  /-
    🎉 no goals
  -/


alias ⟨IsLittleO.of_neg_right, IsLittleO.neg_right⟩ := isLittleO_neg_right


@[simp]
theorem isBigOWith_neg_left : IsBigOWith c l (fun x => -f' x) g ↔ IsBigOWith c l f' g := by
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    inst✝¹ : Norm F
    inst✝ : SeminormedAddCommGroup E'
    c : Real
    g : α → F
    f' : α → E'
    l : Filter α
    ⊢ Iff (Asymptotics.IsBigOWith c l (fun x => Neg.neg (f' x)) g) (Asymptotics.Is …
  -/
  simp only [IsBigOWith_def, norm_neg]
  /-
    🎉 no goals
  -/


alias ⟨IsBigOWith.of_neg_left, IsBigOWith.neg_left⟩ := isBigOWith_neg_left


@[simp]
theorem isBigO_neg_left : (fun x => -f' x) =O[l] g ↔ f' =O[l] g := by
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    inst✝¹ : Norm F
    inst✝ : SeminormedAddCommGroup E'
    g : α → F
    f' : α → E'
    l : Filter α
    ⊢ Iff (Asymptotics.IsBigO l (fun x => Neg.neg (f' x)) g) (Asymptotics.IsBigO l …
  -/
  simp only [IsBigO_def]
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    inst✝¹ : Norm F
    inst✝ : SeminormedAddCommGroup E'
    g : α → F
    f' : α → E'
    l : Filter α
    ⊢ Iff (Exists fun c => Asymptotics.IsBigOWith c l (fun x => Neg.neg (f' x)) g) …
  -/
  exact exists_congr fun _ => isBigOWith_neg_left
  /-
    🎉 no goals
  -/


alias ⟨IsBigO.of_neg_left, IsBigO.neg_left⟩ := isBigO_neg_left


@[simp]
theorem isLittleO_neg_left : (fun x => -f' x) =o[l] g ↔ f' =o[l] g := by
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    inst✝¹ : Norm F
    inst✝ : SeminormedAddCommGroup E'
    g : α → F
    f' : α → E'
    l : Filter α
    ⊢ Iff (Asymptotics.IsLittleO l (fun x => Neg.neg (f' x)) g) (Asymptotics.IsLit …
  -/
  simp only [IsLittleO_def]
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    inst✝¹ : Norm F
    inst✝ : SeminormedAddCommGroup E'
    g : α → F
    f' : α → E'
    l : Filter α
    ⊢ Iff (∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l (fun x => Neg.neg  …
  -/
  exact forall₂_congr fun _ _ => isBigOWith_neg_left
  /-
    🎉 no goals
  -/


alias ⟨IsLittleO.of_neg_left, IsLittleO.neg_left⟩ := isLittleO_neg_left


theorem isBigOWith_fst_prod : IsBigOWith 1 l f' fun x => (f' x, g' x) :=
  isBigOWith_of_le l fun _x => le_max_left _ _


theorem isBigOWith_snd_prod : IsBigOWith 1 l g' fun x => (f' x, g' x) :=
  isBigOWith_of_le l fun _x => le_max_right _ _


theorem isBigO_fst_prod : f' =O[l] fun x => (f' x, g' x) :=
  isBigOWith_fst_prod.isBigO


theorem isBigO_snd_prod : g' =O[l] fun x => (f' x, g' x) :=
  isBigOWith_snd_prod.isBigO


theorem isBigO_fst_prod' {f' : α → E' × F'} : (fun x => (f' x).1) =O[l] f' := by
  /-
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    inst✝¹ : SeminormedAddCommGroup E'
    inst✝ : SeminormedAddCommGroup F'
    l : Filter α
    f' : α → Prod E' F'
    ⊢ Asymptotics.IsBigO l (fun x => (f' x).1) f'
  -/
  simpa [IsBigO_def, IsBigOWith_def] using isBigO_fst_prod (E' := E') (F' := F')
  /-
    🎉 no goals
  -/


theorem isBigO_snd_prod' {f' : α → E' × F'} : (fun x => (f' x).2) =O[l] f' := by
  /-
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    inst✝¹ : SeminormedAddCommGroup E'
    inst✝ : SeminormedAddCommGroup F'
    l : Filter α
    f' : α → Prod E' F'
    ⊢ Asymptotics.IsBigO l (fun x => (f' x).2) f'
  -/
  simpa [IsBigO_def, IsBigOWith_def] using isBigO_snd_prod (E' := E') (F' := F')
  /-
    🎉 no goals
  -/


theorem IsBigOWith.prod_rightl (h : IsBigOWith c l f g') (hc : 0 ≤ c) :
    IsBigOWith c l f fun x => (g' x, k' x) :=
  (h.trans isBigOWith_fst_prod hc).congr_const (mul_one c)


theorem IsBigO.prod_rightl (h : f =O[l] g') : f =O[l] fun x => (g' x, k' x) :=
  let ⟨_c, cnonneg, hc⟩ := h.exists_nonneg
  (hc.prod_rightl k' cnonneg).isBigO


theorem IsLittleO.prod_rightl (h : f =o[l] g') : f =o[l] fun x => (g' x, k' x) :=
  IsLittleO.of_isBigOWith fun _c cpos => (h.forall_isBigOWith cpos).prod_rightl k' cpos.le


theorem IsBigOWith.prod_rightr (h : IsBigOWith c l f g') (hc : 0 ≤ c) :
    IsBigOWith c l f fun x => (f' x, g' x) :=
  (h.trans isBigOWith_snd_prod hc).congr_const (mul_one c)


theorem IsBigO.prod_rightr (h : f =O[l] g') : f =O[l] fun x => (f' x, g' x) :=
  let ⟨_c, cnonneg, hc⟩ := h.exists_nonneg
  (hc.prod_rightr f' cnonneg).isBigO


theorem IsLittleO.prod_rightr (h : f =o[l] g') : f =o[l] fun x => (f' x, g' x) :=
  IsLittleO.of_isBigOWith fun _c cpos => (h.forall_isBigOWith cpos).prod_rightr f' cpos.le


protected theorem IsBigO.fiberwise_right :
    f =O[l ×ˢ l'] g → ∀ᶠ a in l, (f ⟨a, ·⟩) =O[l'] (g ⟨a, ·⟩) := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    l : Filter α
    f : Prod α β → E
    g : Prod α β → F
    l' : Filter β
    ⊢ Asymptotics.IsBigO (SProd.sprod l l') f g → Filter.Eventually (fun a => Asym …
  -/
  simp only [isBigO_iff, eventually_iff, mem_prod_iff]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    l : Filter α
    f : Prod α β → E
    g : Prod α β → F
    l' : Filter β
    ⊢ (Exists fun c => Exists fun t₁ => And (Membership.mem l t₁) (Exists fun t₂ = …
  -/
  rintro ⟨c, t₁, ht₁, t₂, ht₂, ht⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    l : Filter α
    f : Prod α β → E
    g : Prod α β → F
    l' : Filter β
    c : Real
    t₁ : Set α
    ht₁ : Membership.mem l t₁
    t₂ : Set β
    ht₂ : Membership.mem l' t₂
    ht : HasSubset.Subset (SProd.sprod t₁ t₂) (setOf fun x => LE.le (Norm.norm (f  …
    ⊢ Membership.mem l (setOf fun a => Exists fun c => Membership.mem l' (setOf fu …
  -/
  exact mem_of_superset ht₁ fun _ ha ↦ ⟨c, mem_of_superset ht₂ fun _ hb ↦ ht ⟨ha, hb⟩⟩
  /-
    🎉 no goals
  -/


protected theorem IsBigO.fiberwise_left :
    f =O[l ×ˢ l'] g → ∀ᶠ b in l', (f ⟨·, b⟩) =O[l] (g ⟨·, b⟩) := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    l : Filter α
    f : Prod α β → E
    g : Prod α β → F
    l' : Filter β
    ⊢ Asymptotics.IsBigO (SProd.sprod l l') f g → Filter.Eventually (fun b => Asym …
  -/
  simp only [isBigO_iff, eventually_iff, mem_prod_iff]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    l : Filter α
    f : Prod α β → E
    g : Prod α β → F
    l' : Filter β
    ⊢ (Exists fun c => Exists fun t₁ => And (Membership.mem l t₁) (Exists fun t₂ = …
  -/
  rintro ⟨c, t₁, ht₁, t₂, ht₂, ht⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    l : Filter α
    f : Prod α β → E
    g : Prod α β → F
    l' : Filter β
    c : Real
    t₁ : Set α
    ht₁ : Membership.mem l t₁
    t₂ : Set β
    ht₂ : Membership.mem l' t₂
    ht : HasSubset.Subset (SProd.sprod t₁ t₂) (setOf fun x => LE.le (Norm.norm (f  …
    ⊢ Membership.mem l' (setOf fun b => Exists fun c => Membership.mem l (setOf fu …
  -/
  exact mem_of_superset ht₂ fun _ hb ↦ ⟨c, mem_of_superset ht₁ fun _ ha ↦ ht ⟨ha, hb⟩⟩
  /-
    🎉 no goals
  -/


protected theorem IsBigO.comp_fst : f =O[l] g → (f ∘ Prod.fst) =O[l ×ˢ l'] (g ∘ Prod.fst) := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    l : Filter α
    l' : Filter β
    ⊢ Asymptotics.IsBigO l f g → Asymptotics.IsBigO (SProd.sprod l l') (Function.c …
  -/
  simp only [isBigO_iff, eventually_prod_iff]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    l : Filter α
    l' : Filter β
    ⊢ (Exists fun c => Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.h …
  -/
  exact fun ⟨c, hc⟩ ↦ ⟨c, _, hc, fun _ ↦ True, eventually_true l', fun {_} h {_} _ ↦ h⟩
  /-
    🎉 no goals
  -/


protected theorem IsBigO.comp_snd : f =O[l] g → (f ∘ Prod.snd) =O[l' ×ˢ l] (g ∘ Prod.snd) := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    l : Filter α
    l' : Filter β
    ⊢ Asymptotics.IsBigO l f g → Asymptotics.IsBigO (SProd.sprod l' l) (Function.c …
  -/
  simp only [isBigO_iff, eventually_prod_iff]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    l : Filter α
    l' : Filter β
    ⊢ (Exists fun c => Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (HMul.h …
  -/
  exact fun ⟨c, hc⟩ ↦ ⟨c, fun _ ↦ True, eventually_true l', _, hc, fun _ ↦ id⟩
  /-
    🎉 no goals
  -/


protected theorem IsLittleO.comp_fst : f =o[l] g → (f ∘ Prod.fst) =o[l ×ˢ l'] (g ∘ Prod.fst) := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    l : Filter α
    l' : Filter β
    ⊢ Asymptotics.IsLittleO l f g → Asymptotics.IsLittleO (SProd.sprod l l') (Func …
  -/
  simp only [isLittleO_iff, eventually_prod_iff]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    l : Filter α
    l' : Filter β
    ⊢ (∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm (f x …
  -/
  exact fun h _ hc ↦ ⟨_, h hc, fun _ ↦ True, eventually_true l', fun {_} h {_} _ ↦ h⟩
  /-
    🎉 no goals
  -/


protected theorem IsLittleO.comp_snd : f =o[l] g → (f ∘ Prod.snd) =o[l' ×ˢ l] (g ∘ Prod.snd) := by
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    l : Filter α
    l' : Filter β
    ⊢ Asymptotics.IsLittleO l f g → Asymptotics.IsLittleO (SProd.sprod l' l) (Func …
  -/
  simp only [isLittleO_iff, eventually_prod_iff]
  /-
    α : Type u_1
    β : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    l : Filter α
    l' : Filter β
    ⊢ (∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm (f x …
  -/
  exact fun h _ hc ↦ ⟨fun _ ↦ True, eventually_true l', _, h hc, fun _ ↦ id⟩
  /-
    🎉 no goals
  -/


theorem IsBigOWith.prod_left_same (hf : IsBigOWith c l f' k') (hg : IsBigOWith c l g' k') :
    IsBigOWith c l (fun x => (f' x, g' x)) k' := by
  /-
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    G' : Type u_8
    inst✝² : SeminormedAddCommGroup E'
    inst✝¹ : SeminormedAddCommGroup F'
    inst✝ : SeminormedAddCommGroup G'
    c : Real
    f' : α → E'
    g' : α → F'
    k' : α → G'
    l : Filter α
    hf : Asymptotics.IsBigOWith c l f' k'
    hg : Asymptotics.IsBigOWith c l g' k'
    ⊢ Asymptotics.IsBigOWith c l (fun x => { fst := f' x, snd := g' x }) k'
  -/
  rw [isBigOWith_iff] at *; filter_upwards [hf, hg] with x using max_le
                            /-
                              🎉 no goals
                            -/


theorem IsBigOWith.prod_left (hf : IsBigOWith c l f' k') (hg : IsBigOWith c' l g' k') :
    IsBigOWith (max c c') l (fun x => (f' x, g' x)) k' :=
  (hf.weaken <| le_max_left c c').prod_left_same (hg.weaken <| le_max_right c c')


theorem IsBigOWith.prod_left_fst (h : IsBigOWith c l (fun x => (f' x, g' x)) k') :
    IsBigOWith c l f' k' :=
  (isBigOWith_fst_prod.trans h zero_le_one).congr_const <| one_mul c


theorem IsBigOWith.prod_left_snd (h : IsBigOWith c l (fun x => (f' x, g' x)) k') :
    IsBigOWith c l g' k' :=
  (isBigOWith_snd_prod.trans h zero_le_one).congr_const <| one_mul c


theorem isBigOWith_prod_left :
    IsBigOWith c l (fun x => (f' x, g' x)) k' ↔ IsBigOWith c l f' k' ∧ IsBigOWith c l g' k' :=
  ⟨fun h => ⟨h.prod_left_fst, h.prod_left_snd⟩, fun h => h.1.prod_left_same h.2⟩


theorem IsBigO.prod_left (hf : f' =O[l] k') (hg : g' =O[l] k') : (fun x => (f' x, g' x)) =O[l] k' :=
  let ⟨_c, hf⟩ := hf.isBigOWith
  let ⟨_c', hg⟩ := hg.isBigOWith
  (hf.prod_left hg).isBigO


theorem IsBigO.prod_left_fst : (fun x => (f' x, g' x)) =O[l] k' → f' =O[l] k' :=
  IsBigO.trans isBigO_fst_prod


theorem IsBigO.prod_left_snd : (fun x => (f' x, g' x)) =O[l] k' → g' =O[l] k' :=
  IsBigO.trans isBigO_snd_prod


@[simp]
theorem isBigO_prod_left : (fun x => (f' x, g' x)) =O[l] k' ↔ f' =O[l] k' ∧ g' =O[l] k' :=
  ⟨fun h => ⟨h.prod_left_fst, h.prod_left_snd⟩, fun h => h.1.prod_left h.2⟩


theorem IsLittleO.prod_left (hf : f' =o[l] k') (hg : g' =o[l] k') :
    (fun x => (f' x, g' x)) =o[l] k' :=
  IsLittleO.of_isBigOWith fun _c hc =>
    (hf.forall_isBigOWith hc).prod_left_same (hg.forall_isBigOWith hc)


theorem IsLittleO.prod_left_fst : (fun x => (f' x, g' x)) =o[l] k' → f' =o[l] k' :=
  IsBigO.trans_isLittleO isBigO_fst_prod


theorem IsLittleO.prod_left_snd : (fun x => (f' x, g' x)) =o[l] k' → g' =o[l] k' :=
  IsBigO.trans_isLittleO isBigO_snd_prod


@[simp]
theorem isLittleO_prod_left : (fun x => (f' x, g' x)) =o[l] k' ↔ f' =o[l] k' ∧ g' =o[l] k' :=
  ⟨fun h => ⟨h.prod_left_fst, h.prod_left_snd⟩, fun h => h.1.prod_left h.2⟩


theorem IsBigOWith.eq_zero_imp (h : IsBigOWith c l f'' g'') : ∀ᶠ x in l, g'' x = 0 → f'' x = 0 :=
                                                                  /-
                                                                    α : Type u_1
                                                                    E'' : Type u_9
                                                                    F'' : Type u_10
                                                                    inst✝¹ : NormedAddCommGroup E''
                                                                    inst✝ : NormedAddCommGroup F''
                                                                    c : Real
                                                                    f'' : α → E''
                                                                    g'' : α → F''
                                                                    l : Filter α
                                                                    h : Asymptotics.IsBigOWith c l f'' g''
                                                                    x : α
                                                                    hx : LE.le (Norm.norm (f'' x)) (HMul.hMul c (Norm.norm (g'' x)))
                                                                    hg : Eq (g'' x) 0
                                                                    ⊢ LE.le (Norm.norm (f'' x)) 0
                                                                  -/
  Eventually.mono h.bound fun x hx hg => norm_le_zero_iff.1 <| by simpa [hg] using hx
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem IsBigO.eq_zero_imp (h : f'' =O[l] g'') : ∀ᶠ x in l, g'' x = 0 → f'' x = 0 :=
  let ⟨_C, hC⟩ := h.isBigOWith
  hC.eq_zero_imp


theorem IsBigOWith.add (h₁ : IsBigOWith c₁ l f₁ g) (h₂ : IsBigOWith c₂ l f₂ g) :
    IsBigOWith (c₁ + c₂) l (fun x => f₁ x + f₂ x) g := by
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    inst✝¹ : Norm F
    inst✝ : SeminormedAddCommGroup E'
    c₁ c₂ : Real
    g : α → F
    l : Filter α
    f₁ f₂ : α → E'
    h₁ : Asymptotics.IsBigOWith c₁ l f₁ g
    h₂ : Asymptotics.IsBigOWith c₂ l f₂ g
    ⊢ Asymptotics.IsBigOWith (HAdd.hAdd c₁ c₂) l (fun x => HAdd.hAdd (f₁ x) (f₂ x) …
  -/
  rw [IsBigOWith_def] at *
  filter_upwards [h₁, h₂] with x hx₁ hx₂ using
    calc
      ‖f₁ x + f₂ x‖ ≤ c₁ * ‖g x‖ + c₂ * ‖g x‖ := norm_add_le_of_le hx₁ hx₂
      _ = (c₁ + c₂) * ‖g x‖ := (add_mul _ _ _).symm


theorem IsBigO.add (h₁ : f₁ =O[l] g) (h₂ : f₂ =O[l] g) : (fun x => f₁ x + f₂ x) =O[l] g :=
  let ⟨_c₁, hc₁⟩ := h₁.isBigOWith
  let ⟨_c₂, hc₂⟩ := h₂.isBigOWith
  (hc₁.add hc₂).isBigO


theorem IsLittleO.add (h₁ : f₁ =o[l] g) (h₂ : f₂ =o[l] g) : (fun x => f₁ x + f₂ x) =o[l] g :=
  IsLittleO.of_isBigOWith fun c cpos =>
    ((h₁.forall_isBigOWith <| half_pos cpos).add (h₂.forall_isBigOWith <|
      half_pos cpos)).congr_const (add_halves c)


theorem IsLittleO.add_add (h₁ : f₁ =o[l] g₁) (h₂ : f₂ =o[l] g₂) :
    (fun x => f₁ x + f₂ x) =o[l] fun x => ‖g₁ x‖ + ‖g₂ x‖ := by
  /-
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    inst✝¹ : SeminormedAddCommGroup E'
    inst✝ : SeminormedAddCommGroup F'
    l : Filter α
    f₁ f₂ : α → E'
    g₁ g₂ : α → F'
    h₁ : Asymptotics.IsLittleO l f₁ g₁
    h₂ : Asymptotics.IsLittleO l f₂ g₂
    ⊢ Asymptotics.IsLittleO l (fun x => HAdd.hAdd (f₁ x) (f₂ x)) fun x => HAdd.hAd …
  -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  refine (h₁.trans_le fun x => ?_).add (h₂.trans_le ?_) <;> simp [abs_of_nonneg, add_nonneg]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem IsBigO.add_isLittleO (h₁ : f₁ =O[l] g) (h₂ : f₂ =o[l] g) : (fun x => f₁ x + f₂ x) =O[l] g :=
  h₁.add h₂.isBigO


theorem IsLittleO.add_isBigO (h₁ : f₁ =o[l] g) (h₂ : f₂ =O[l] g) : (fun x => f₁ x + f₂ x) =O[l] g :=
  h₁.isBigO.add h₂


theorem IsBigOWith.add_isLittleO (h₁ : IsBigOWith c₁ l f₁ g) (h₂ : f₂ =o[l] g) (hc : c₁ < c₂) :
    IsBigOWith c₂ l (fun x => f₁ x + f₂ x) g :=
  (h₁.add (h₂.forall_isBigOWith (sub_pos.2 hc))).congr_const (add_sub_cancel _ _)


theorem IsLittleO.add_isBigOWith (h₁ : f₁ =o[l] g) (h₂ : IsBigOWith c₁ l f₂ g) (hc : c₁ < c₂) :
    IsBigOWith c₂ l (fun x => f₁ x + f₂ x) g :=
  (h₂.add_isLittleO h₁ hc).congr_left fun _ => add_comm _ _


theorem IsBigOWith.sub (h₁ : IsBigOWith c₁ l f₁ g) (h₂ : IsBigOWith c₂ l f₂ g) :
    IsBigOWith (c₁ + c₂) l (fun x => f₁ x - f₂ x) g := by
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    inst✝¹ : Norm F
    inst✝ : SeminormedAddCommGroup E'
    c₁ c₂ : Real
    g : α → F
    l : Filter α
    f₁ f₂ : α → E'
    h₁ : Asymptotics.IsBigOWith c₁ l f₁ g
    h₂ : Asymptotics.IsBigOWith c₂ l f₂ g
    ⊢ Asymptotics.IsBigOWith (HAdd.hAdd c₁ c₂) l (fun x => HSub.hSub (f₁ x) (f₂ x) …
  -/
  simpa only [sub_eq_add_neg] using h₁.add h₂.neg_left
  /-
    🎉 no goals
  -/


theorem IsBigOWith.sub_isLittleO (h₁ : IsBigOWith c₁ l f₁ g) (h₂ : f₂ =o[l] g) (hc : c₁ < c₂) :
    IsBigOWith c₂ l (fun x => f₁ x - f₂ x) g := by
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    inst✝¹ : Norm F
    inst✝ : SeminormedAddCommGroup E'
    c₁ c₂ : Real
    g : α → F
    l : Filter α
    f₁ f₂ : α → E'
    h₁ : Asymptotics.IsBigOWith c₁ l f₁ g
    h₂ : Asymptotics.IsLittleO l f₂ g
    hc : LT.lt c₁ c₂
    ⊢ Asymptotics.IsBigOWith c₂ l (fun x => HSub.hSub (f₁ x) (f₂ x)) g
  -/
  simpa only [sub_eq_add_neg] using h₁.add_isLittleO h₂.neg_left hc
  /-
    🎉 no goals
  -/


theorem IsBigO.sub (h₁ : f₁ =O[l] g) (h₂ : f₂ =O[l] g) : (fun x => f₁ x - f₂ x) =O[l] g := by
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    inst✝¹ : Norm F
    inst✝ : SeminormedAddCommGroup E'
    g : α → F
    l : Filter α
    f₁ f₂ : α → E'
    h₁ : Asymptotics.IsBigO l f₁ g
    h₂ : Asymptotics.IsBigO l f₂ g
    ⊢ Asymptotics.IsBigO l (fun x => HSub.hSub (f₁ x) (f₂ x)) g
  -/
  simpa only [sub_eq_add_neg] using h₁.add h₂.neg_left
  /-
    🎉 no goals
  -/


theorem IsLittleO.sub (h₁ : f₁ =o[l] g) (h₂ : f₂ =o[l] g) : (fun x => f₁ x - f₂ x) =o[l] g := by
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    inst✝¹ : Norm F
    inst✝ : SeminormedAddCommGroup E'
    g : α → F
    l : Filter α
    f₁ f₂ : α → E'
    h₁ : Asymptotics.IsLittleO l f₁ g
    h₂ : Asymptotics.IsLittleO l f₂ g
    ⊢ Asymptotics.IsLittleO l (fun x => HSub.hSub (f₁ x) (f₂ x)) g
  -/
  simpa only [sub_eq_add_neg] using h₁.add h₂.neg_left
  /-
    🎉 no goals
  -/


theorem IsBigOWith.symm (h : IsBigOWith c l (fun x => f₁ x - f₂ x) g) :
    IsBigOWith c l (fun x => f₂ x - f₁ x) g :=
  h.neg_left.congr_left fun _x => neg_sub _ _


theorem isBigOWith_comm :
    IsBigOWith c l (fun x => f₁ x - f₂ x) g ↔ IsBigOWith c l (fun x => f₂ x - f₁ x) g :=
  ⟨IsBigOWith.symm, IsBigOWith.symm⟩


theorem IsBigO.symm (h : (fun x => f₁ x - f₂ x) =O[l] g) : (fun x => f₂ x - f₁ x) =O[l] g :=
  h.neg_left.congr_left fun _x => neg_sub _ _


theorem isBigO_comm : (fun x => f₁ x - f₂ x) =O[l] g ↔ (fun x => f₂ x - f₁ x) =O[l] g :=
  ⟨IsBigO.symm, IsBigO.symm⟩


theorem IsLittleO.symm (h : (fun x => f₁ x - f₂ x) =o[l] g) : (fun x => f₂ x - f₁ x) =o[l] g := by
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    inst✝¹ : Norm F
    inst✝ : SeminormedAddCommGroup E'
    g : α → F
    l : Filter α
    f₁ f₂ : α → E'
    h : Asymptotics.IsLittleO l (fun x => HSub.hSub (f₁ x) (f₂ x)) g
    ⊢ Asymptotics.IsLittleO l (fun x => HSub.hSub (f₂ x) (f₁ x)) g
  -/
  simpa only [neg_sub] using h.neg_left
  /-
    🎉 no goals
  -/


theorem isLittleO_comm : (fun x => f₁ x - f₂ x) =o[l] g ↔ (fun x => f₂ x - f₁ x) =o[l] g :=
  ⟨IsLittleO.symm, IsLittleO.symm⟩


theorem IsBigOWith.triangle (h₁ : IsBigOWith c l (fun x => f₁ x - f₂ x) g)
    (h₂ : IsBigOWith c' l (fun x => f₂ x - f₃ x) g) :
    IsBigOWith (c + c') l (fun x => f₁ x - f₃ x) g :=
  (h₁.add h₂).congr_left fun _x => sub_add_sub_cancel _ _ _


theorem IsBigO.triangle (h₁ : (fun x => f₁ x - f₂ x) =O[l] g)
    (h₂ : (fun x => f₂ x - f₃ x) =O[l] g) : (fun x => f₁ x - f₃ x) =O[l] g :=
  (h₁.add h₂).congr_left fun _x => sub_add_sub_cancel _ _ _


theorem IsLittleO.triangle (h₁ : (fun x => f₁ x - f₂ x) =o[l] g)
    (h₂ : (fun x => f₂ x - f₃ x) =o[l] g) : (fun x => f₁ x - f₃ x) =o[l] g :=
  (h₁.add h₂).congr_left fun _x => sub_add_sub_cancel _ _ _


theorem IsBigO.congr_of_sub (h : (fun x => f₁ x - f₂ x) =O[l] g) : f₁ =O[l] g ↔ f₂ =O[l] g :=
  ⟨fun h' => (h'.sub h).congr_left fun _x => sub_sub_cancel _ _, fun h' =>
    (h.add h').congr_left fun _x => sub_add_cancel _ _⟩


theorem IsLittleO.congr_of_sub (h : (fun x => f₁ x - f₂ x) =o[l] g) : f₁ =o[l] g ↔ f₂ =o[l] g :=
  ⟨fun h' => (h'.sub h).congr_left fun _x => sub_sub_cancel _ _, fun h' =>
    (h.add h').congr_left fun _x => sub_add_cancel _ _⟩


theorem isLittleO_zero : (fun _x => (0 : E')) =o[l] g' :=
  IsLittleO.of_bound fun c hc =>
                          /-
                            α : Type u_1
                            E' : Type u_6
                            F' : Type u_7
                            inst✝¹ : SeminormedAddCommGroup E'
                            inst✝ : SeminormedAddCommGroup F'
                            g' : α → F'
                            l : Filter α
                            c : Real
                            hc : LT.lt 0 c
                            x : α
                            ⊢ Membership.mem (setOf fun x => (fun x => LE.le (Norm.norm 0) (HMul.hMul c (N …
                          -/
    univ_mem' fun x => by simpa using mul_nonneg hc.le (norm_nonneg <| g' x)
                          /-
                            🎉 no goals
                          -/


theorem isBigOWith_zero (hc : 0 ≤ c) : IsBigOWith c l (fun _x => (0 : E')) g' :=
                                               /-
                                                 α : Type u_1
                                                 E' : Type u_6
                                                 F' : Type u_7
                                                 inst✝¹ : SeminormedAddCommGroup E'
                                                 inst✝ : SeminormedAddCommGroup F'
                                                 c : Real
                                                 g' : α → F'
                                                 l : Filter α
                                                 hc : LE.le 0 c
                                                 x : α
                                                 ⊢ Membership.mem (setOf fun x => (fun x => LE.le (Norm.norm 0) (HMul.hMul c (N …
                                               -/
  IsBigOWith.of_bound <| univ_mem' fun x => by simpa using mul_nonneg hc (norm_nonneg <| g' x)
                                               /-
                                                 🎉 no goals
                                               -/


theorem isBigOWith_zero' : IsBigOWith 0 l (fun _x => (0 : E')) g :=
                                               /-
                                                 α : Type u_1
                                                 F : Type u_4
                                                 E' : Type u_6
                                                 inst✝¹ : Norm F
                                                 inst✝ : SeminormedAddCommGroup E'
                                                 g : α → F
                                                 l : Filter α
                                                 x : α
                                                 ⊢ Membership.mem (setOf fun x => (fun x => LE.le (Norm.norm 0) (HMul.hMul 0 (N …
                                               -/
  IsBigOWith.of_bound <| univ_mem' fun x => by simp
                                               /-
                                                 🎉 no goals
                                               -/


theorem isBigO_zero : (fun _x => (0 : E')) =O[l] g :=
  isBigO_iff_isBigOWith.2 ⟨0, isBigOWith_zero' _ _⟩


theorem isBigO_refl_left : (fun x => f' x - f' x) =O[l] g' :=
  (isBigO_zero g' l).congr_left fun _x => (sub_self _).symm


theorem isLittleO_refl_left : (fun x => f' x - f' x) =o[l] g' :=
  (isLittleO_zero g' l).congr_left fun _x => (sub_self _).symm


@[simp]
theorem isBigOWith_zero_right_iff : (IsBigOWith c l f'' fun _x => (0 : F')) ↔ f'' =ᶠ[l] 0 := by
  simp only [IsBigOWith_def, exists_prop, norm_zero, mul_zero,
    norm_le_zero_iff, EventuallyEq, Pi.zero_apply]


@[simp]
theorem isBigO_zero_right_iff : (f'' =O[l] fun _x => (0 : F')) ↔ f'' =ᶠ[l] 0 :=
  ⟨fun h =>
    let ⟨_c, hc⟩ := h.isBigOWith
    isBigOWith_zero_right_iff.1 hc,
    fun h => (isBigOWith_zero_right_iff.2 h : IsBigOWith 1 _ _ _).isBigO⟩


@[simp]
theorem isLittleO_zero_right_iff : (f'' =o[l] fun _x => (0 : F')) ↔ f'' =ᶠ[l] 0 :=
  ⟨fun h => isBigO_zero_right_iff.1 h.isBigO,
   fun h => IsLittleO.of_isBigOWith fun _c _hc => isBigOWith_zero_right_iff.2 h⟩


theorem isBigOWith_const_const (c : E) {c' : F''} (hc' : c' ≠ 0) (l : Filter α) :
    IsBigOWith (‖c‖ / ‖c'‖) l (fun _x : α => c) fun _x => c' := by
  /-
    α : Type u_1
    E : Type u_3
    F'' : Type u_10
    inst✝¹ : Norm E
    inst✝ : NormedAddCommGroup F''
    c : E
    c' : F''
    hc' : Ne c' 0
    l : Filter α
    ⊢ Asymptotics.IsBigOWith (HDiv.hDiv (Norm.norm c) (Norm.norm c')) l (fun _x => …
  -/
  simp only [IsBigOWith_def]
  /-
    α : Type u_1
    E : Type u_3
    F'' : Type u_10
    inst✝¹ : Norm E
    inst✝ : NormedAddCommGroup F''
    c : E
    c' : F''
    hc' : Ne c' 0
    l : Filter α
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm c) (HMul.hMul (HDiv.hDiv (Norm. …
  -/
  apply univ_mem'
  /-
    case h
    α : Type u_1
    E : Type u_3
    F'' : Type u_10
    inst✝¹ : Norm E
    inst✝ : NormedAddCommGroup F''
    c : E
    c' : F''
    hc' : Ne c' 0
    l : Filter α
    ⊢ ∀ (a : α), Membership.mem (setOf fun x => (fun x => LE.le (Norm.norm c) (HMu …
  -/
  intro x
  /-
    case h
    α : Type u_1
    E : Type u_3
    F'' : Type u_10
    inst✝¹ : Norm E
    inst✝ : NormedAddCommGroup F''
    c : E
    c' : F''
    hc' : Ne c' 0
    l : Filter α
    x : α
    ⊢ Membership.mem (setOf fun x => (fun x => LE.le (Norm.norm c) (HMul.hMul (HDi …
  -/
  rw [mem_setOf, div_mul_cancel₀ _ (norm_ne_zero_iff.mpr hc')]
  /-
    🎉 no goals
  -/


theorem isBigO_const_const (c : E) {c' : F''} (hc' : c' ≠ 0) (l : Filter α) :
    (fun _x : α => c) =O[l] fun _x => c' :=
  (isBigOWith_const_const c hc' l).isBigO


@[simp]
theorem isBigO_const_const_iff {c : E''} {c' : F''} (l : Filter α) [l.NeBot] :
    ((fun _x : α => c) =O[l] fun _x => c') ↔ c' = 0 → c = 0 := by
  /-
    α : Type u_1
    E'' : Type u_9
    F'' : Type u_10
    inst✝² : NormedAddCommGroup E''
    inst✝¹ : NormedAddCommGroup F''
    c : E''
    c' : F''
    l : Filter α
    inst✝ : l.NeBot
    ⊢ Iff (Asymptotics.IsBigO l (fun _x => c) fun _x => c') (Eq c' 0 → Eq c 0)
  -/
  rcases eq_or_ne c' 0 with (rfl | hc')
    /-
      case inl
      α : Type u_1
      E'' : Type u_9
      F'' : Type u_10
      inst✝² : NormedAddCommGroup E''
      inst✝¹ : NormedAddCommGroup F''
      c : E''
      l : Filter α
      inst✝ : l.NeBot
      ⊢ Iff (Asymptotics.IsBigO l (fun _x => c) fun _x => 0) (Eq 0 0 → Eq c 0)
    -/
  · simp [EventuallyEq]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      E'' : Type u_9
      F'' : Type u_10
      inst✝² : NormedAddCommGroup E''
      inst✝¹ : NormedAddCommGroup F''
      c : E''
      c' : F''
      l : Filter α
      inst✝ : l.NeBot
      hc' : Ne c' 0
      ⊢ Iff (Asymptotics.IsBigO l (fun _x => c) fun _x => c') (Eq c' 0 → Eq c 0)
    -/
  · simp [hc', isBigO_const_const _ hc']
    /-
      🎉 no goals
    -/


@[simp]
theorem isBigO_pure {x} : f'' =O[pure x] g'' ↔ g'' x = 0 → f'' x = 0 :=
  calc
    f'' =O[pure x] g'' ↔ (fun _y : α => f'' x) =O[pure x] fun _ => g'' x := isBigO_congr rfl rfl
    _ ↔ g'' x = 0 → f'' x = 0 := isBigO_const_const_iff _


@[simp]
theorem isBigOWith_principal {s : Set α} : IsBigOWith c (𝓟 s) f g ↔ ∀ x ∈ s, ‖f x‖ ≤ c * ‖g x‖ := by
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    c : Real
    f : α → E
    g : α → F
    s : Set α
    ⊢ Iff (Asymptotics.IsBigOWith c (Filter.principal s) f g) (∀ (x : α), Membersh …
  -/
  rw [IsBigOWith_def, eventually_principal]
  /-
    🎉 no goals
  -/


theorem isBigO_principal {s : Set α} : f =O[𝓟 s] g ↔ ∃ c, ∀ x ∈ s, ‖f x‖ ≤ c * ‖g x‖ := by
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    s : Set α
    ⊢ Iff (Asymptotics.IsBigO (Filter.principal s) f g) (Exists fun c => ∀ (x : α) …
  -/
  simp_rw [isBigO_iff, eventually_principal]
  /-
    🎉 no goals
  -/


@[simp]
theorem isLittleO_principal {s : Set α} : f'' =o[𝓟 s] g' ↔ ∀ x ∈ s, f'' x = 0 := by
  /-
    α : Type u_1
    F' : Type u_7
    E'' : Type u_9
    inst✝¹ : SeminormedAddCommGroup F'
    inst✝ : NormedAddCommGroup E''
    g' : α → F'
    f'' : α → E''
    s : Set α
    ⊢ Iff (Asymptotics.IsLittleO (Filter.principal s) f'' g') (∀ (x : α), Membersh …
  -/
  refine ⟨fun h x hx ↦ norm_le_zero_iff.1 ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      F' : Type u_7
      E'' : Type u_9
      inst✝¹ : SeminormedAddCommGroup F'
      inst✝ : NormedAddCommGroup E''
      g' : α → F'
      f'' : α → E''
      s : Set α
      h : Asymptotics.IsLittleO (Filter.principal s) f'' g'
      x : α
      hx : Membership.mem s x
      ⊢ LE.le (Norm.norm (f'' x)) 0
    -/
  · simp only [isLittleO_iff, isBigOWith_principal] at h
    have : Tendsto (fun c : ℝ => c * ‖g' x‖) (𝓝[>] 0) (𝓝 0) :=
      ((continuous_id.mul continuous_const).tendsto' _ _ (zero_mul _)).mono_left
        inf_le_left
    /-
      case refine_1
      α : Type u_1
      F' : Type u_7
      E'' : Type u_9
      inst✝¹ : SeminormedAddCommGroup F'
      inst✝ : NormedAddCommGroup E''
      g' : α → F'
      f'' : α → E''
      s : Set α
      x : α
      hx : Membership.mem s x
      h : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm (f' …
      this : Filter.Tendsto (fun c => HMul.hMul c (Norm.norm (g' x))) (nhdsWithin 0  …
      ⊢ LE.le (Norm.norm (f'' x)) 0
    -/
    apply le_of_tendsto_of_tendsto tendsto_const_nhds this
    /-
      case refine_1
      α : Type u_1
      F' : Type u_7
      E'' : Type u_9
      inst✝¹ : SeminormedAddCommGroup F'
      inst✝ : NormedAddCommGroup E''
      g' : α → F'
      f'' : α → E''
      s : Set α
      x : α
      hx : Membership.mem s x
      h : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm (f' …
      this : Filter.Tendsto (fun c => HMul.hMul c (Norm.norm (g' x))) (nhdsWithin 0  …
      ⊢ (nhdsWithin 0 (Set.Ioi 0)).EventuallyLE (fun x_1 => Norm.norm (f'' x)) fun c …
    -/
    apply eventually_nhdsWithin_iff.2 (Eventually.of_forall (fun c hc ↦ ?_))
    /-
      α : Type u_1
      F' : Type u_7
      E'' : Type u_9
      inst✝¹ : SeminormedAddCommGroup F'
      inst✝ : NormedAddCommGroup E''
      g' : α → F'
      f'' : α → E''
      s : Set α
      x : α
      hx : Membership.mem s x
      h : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm (f' …
      this : Filter.Tendsto (fun c => HMul.hMul c (Norm.norm (g' x))) (nhdsWithin 0  …
      c : Real
      hc : Membership.mem (Set.Ioi 0) c
      ⊢ LE.le ((fun x_1 => Norm.norm (f'' x)) c) ((fun c => HMul.hMul c (Norm.norm ( …
    -/
    exact eventually_principal.1 (h hc) x hx
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      F' : Type u_7
      E'' : Type u_9
      inst✝¹ : SeminormedAddCommGroup F'
      inst✝ : NormedAddCommGroup E''
      g' : α → F'
      f'' : α → E''
      s : Set α
      h : ∀ (x : α), Membership.mem s x → Eq (f'' x) 0
      ⊢ Asymptotics.IsLittleO (Filter.principal s) f'' g'
    -/
  · apply (isLittleO_zero g' _).congr' ?_ EventuallyEq.rfl
    /-
      α : Type u_1
      F' : Type u_7
      E'' : Type u_9
      inst✝¹ : SeminormedAddCommGroup F'
      inst✝ : NormedAddCommGroup E''
      g' : α → F'
      f'' : α → E''
      s : Set α
      h : ∀ (x : α), Membership.mem s x → Eq (f'' x) 0
      ⊢ (Filter.principal s).EventuallyEq (fun _x => 0) f''
    -/
    exact fun x hx ↦ (h x hx).symm
    /-
      🎉 no goals
    -/


@[simp]
theorem isBigOWith_top : IsBigOWith c ⊤ f g ↔ ∀ x, ‖f x‖ ≤ c * ‖g x‖ := by
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    c : Real
    f : α → E
    g : α → F
    ⊢ Iff (Asymptotics.IsBigOWith c Top.top f g) (∀ (x : α), LE.le (Norm.norm (f x …
  -/
  rw [IsBigOWith_def, eventually_top]
  /-
    🎉 no goals
  -/


@[simp]
theorem isBigO_top : f =O[⊤] g ↔ ∃ C, ∀ x, ‖f x‖ ≤ C * ‖g x‖ := by
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝¹ : Norm E
    inst✝ : Norm F
    f : α → E
    g : α → F
    ⊢ Iff (Asymptotics.IsBigO Top.top f g) (Exists fun C => ∀ (x : α), LE.le (Norm …
  -/
  simp_rw [isBigO_iff, eventually_top]
  /-
    🎉 no goals
  -/


@[simp]
theorem isLittleO_top : f'' =o[⊤] g' ↔ ∀ x, f'' x = 0 := by
  /-
    α : Type u_1
    F' : Type u_7
    E'' : Type u_9
    inst✝¹ : SeminormedAddCommGroup F'
    inst✝ : NormedAddCommGroup E''
    g' : α → F'
    f'' : α → E''
    ⊢ Iff (Asymptotics.IsLittleO Top.top f'' g') (∀ (x : α), Eq (f'' x) 0)
  -/
  simp only [← principal_univ, isLittleO_principal, mem_univ, forall_true_left]
  /-
    🎉 no goals
  -/


theorem isBigOWith_const_one (c : E) (l : Filter α) :
                                                               /-
                                                                 α : Type u_1
                                                                 E : Type u_3
                                                                 F : Type u_4
                                                                 inst✝³ : Norm E
                                                                 inst✝² : Norm F
                                                                 inst✝¹ : One F
                                                                 inst✝ : NormOneClass F
                                                                 c : E
                                                                 l : Filter α
                                                                 ⊢ Asymptotics.IsBigOWith (Norm.norm c) l (fun _x => c) fun _x => 1
                                                               -/
    IsBigOWith ‖c‖ l (fun _x : α => c) fun _x => (1 : F) := by simp [isBigOWith_iff]
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem isBigO_const_one (c : E) (l : Filter α) : (fun _x : α => c) =O[l] fun _x => (1 : F) :=
  (isBigOWith_const_one F c l).isBigO


theorem isLittleO_const_iff_isLittleO_one {c : F''} (hc : c ≠ 0) :
    (f =o[l] fun _x => c) ↔ f =o[l] fun _x => (1 : F) :=
  ⟨fun h => h.trans_isBigOWith (isBigOWith_const_one _ _ _) (norm_pos_iff.2 hc),
   fun h => h.trans_isBigO <| isBigO_const_const _ hc _⟩


@[simp]
theorem isLittleO_one_iff {f : α → E'''} : f =o[l] (fun _x => 1 : α → F) ↔ Tendsto f l (𝓝 0) := by
  simp only [isLittleO_iff, norm_one, mul_one, Metric.nhds_basis_closedBall.tendsto_right_iff,
    Metric.mem_closedBall, dist_zero_right]


@[simp]
theorem isBigO_one_iff : f =O[l] (fun _x => 1 : α → F) ↔
    IsBoundedUnder (· ≤ ·) l fun x => ‖f x‖ := by
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝³ : Norm E
    inst✝² : Norm F
    f : α → E
    l : Filter α
    inst✝¹ : One F
    inst✝ : NormOneClass F
    ⊢ Iff (Asymptotics.IsBigO l f fun _x => 1) (Filter.IsBoundedUnder (fun x1 x2 = …
  -/
  simp only [isBigO_iff, norm_one, mul_one, IsBoundedUnder, IsBounded, eventually_map]
  /-
    🎉 no goals
  -/


alias ⟨_, _root_.Filter.IsBoundedUnder.isBigO_one⟩ := isBigO_one_iff


@[simp]
theorem isLittleO_one_left_iff : (fun _x => 1 : α → F) =o[l] f ↔ Tendsto (fun x => ‖f x‖) l atTop :=
  calc
    (fun _x => 1 : α → F) =o[l] f ↔ ∀ n : ℕ, ∀ᶠ x in l, ↑n * ‖(1 : F)‖ ≤ ‖f x‖ :=
                                                          /-
                                                            α : Type u_1
                                                            E : Type u_3
                                                            F : Type u_4
                                                            inst✝³ : Norm E
                                                            inst✝² : Norm F
                                                            f : α → E
                                                            l : Filter α
                                                            inst✝¹ : One F
                                                            inst✝ : NormOneClass F
                                                            _x : α
                                                            ⊢ LE.le 0 (Norm.norm 1)
                                                          -/
      isLittleO_iff_nat_mul_le_aux <| Or.inl fun _x => by simp only [norm_one, zero_le_one]
                                                          /-
                                                            🎉 no goals
                                                          -/
    _ ↔ ∀ n : ℕ, True → ∀ᶠ x in l, ‖f x‖ ∈ Ici (n : ℝ) := by
      /-
        α : Type u_1
        E : Type u_3
        F : Type u_4
        inst✝³ : Norm E
        inst✝² : Norm F
        f : α → E
        l : Filter α
        inst✝¹ : One F
        inst✝ : NormOneClass F
        ⊢ Iff (∀ (n : Nat), Filter.Eventually (fun x => LE.le (HMul.hMul (↑n) (Norm.no …
      -/
      simp only [norm_one, mul_one, true_imp_iff, mem_Ici]
      /-
        🎉 no goals
      -/
    _ ↔ Tendsto (fun x => ‖f x‖) l atTop :=
      atTop_hasCountableBasis_of_archimedean.1.tendsto_right_iff.symm


theorem _root_.Filter.Tendsto.isBigO_one {c : E'} (h : Tendsto f' l (𝓝 c)) :
    f' =O[l] (fun _x => 1 : α → F) :=
  h.norm.isBoundedUnder_le.isBigO_one F


theorem IsBigO.trans_tendsto_nhds (hfg : f =O[l] g') {y : F'} (hg : Tendsto g' l (𝓝 y)) :
    f =O[l] (fun _x => 1 : α → F) :=
  hfg.trans <| hg.isBigO_one F


/-- The condition `f = O[𝓝[≠] a] 1` is equivalent to `f = O[𝓝 a] 1`. -/
lemma isBigO_one_nhds_ne_iff [TopologicalSpace α] {a : α} :
    f =O[𝓝[≠] a] (fun _ ↦ 1 : α → F) ↔ f =O[𝓝 a] (fun _ ↦ 1 : α → F) := by
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁴ : Norm E
    inst✝³ : Norm F
    f : α → E
    inst✝² : One F
    inst✝¹ : NormOneClass F
    inst✝ : TopologicalSpace α
    a : α
    ⊢ Iff (Asymptotics.IsBigO (nhdsWithin a (HasCompl.compl (Singleton.singleton a …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ h.mono nhdsWithin_le_nhds⟩
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁴ : Norm E
    inst✝³ : Norm F
    f : α → E
    inst✝² : One F
    inst✝¹ : NormOneClass F
    inst✝ : TopologicalSpace α
    a : α
    h : Asymptotics.IsBigO (nhdsWithin a (HasCompl.compl (Singleton.singleton a))) …
    ⊢ Asymptotics.IsBigO (nhds a) f fun x => 1
  -/
  simp only [isBigO_one_iff, IsBoundedUnder, IsBounded, eventually_map] at h ⊢
  /-
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁴ : Norm E
    inst✝³ : Norm F
    f : α → E
    inst✝² : One F
    inst✝¹ : NormOneClass F
    inst✝ : TopologicalSpace α
    a : α
    h : Exists fun b => Filter.Eventually (fun a => LE.le (Norm.norm (f a)) b) (nh …
    ⊢ Exists fun b => Filter.Eventually (fun a => LE.le (Norm.norm (f a)) b) (nhds …
  -/
  obtain ⟨c, hc⟩ := h
  /-
    case intro
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁴ : Norm E
    inst✝³ : Norm F
    f : α → E
    inst✝² : One F
    inst✝¹ : NormOneClass F
    inst✝ : TopologicalSpace α
    a : α
    c : Real
    hc : Filter.Eventually (fun a => LE.le (Norm.norm (f a)) c) (nhdsWithin a (Has …
    ⊢ Exists fun b => Filter.Eventually (fun a => LE.le (Norm.norm (f a)) b) (nhds …
  -/
  use max c ‖f a‖
  /-
    case h
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁴ : Norm E
    inst✝³ : Norm F
    f : α → E
    inst✝² : One F
    inst✝¹ : NormOneClass F
    inst✝ : TopologicalSpace α
    a : α
    c : Real
    hc : Filter.Eventually (fun a => LE.le (Norm.norm (f a)) c) (nhdsWithin a (Has …
    ⊢ Filter.Eventually (fun a_1 => LE.le (Norm.norm (f a_1)) (Max.max c (Norm.nor …
  -/
  filter_upwards [eventually_nhdsWithin_iff.mp hc] with b hb
  /-
    case h
    α : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁴ : Norm E
    inst✝³ : Norm F
    f : α → E
    inst✝² : One F
    inst✝¹ : NormOneClass F
    inst✝ : TopologicalSpace α
    a : α
    c : Real
    hc : Filter.Eventually (fun a => LE.le (Norm.norm (f a)) c) (nhdsWithin a (Has …
    b : α
    hb : Membership.mem (HasCompl.compl (Singleton.singleton a)) b → LE.le (Norm.n …
    ⊢ LE.le (Norm.norm (f b)) (Max.max c (Norm.norm (f a)))
  -/
  rcases eq_or_ne b a with rfl | hb'
    /-
      case h.inl
      α : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝⁴ : Norm E
      inst✝³ : Norm F
      f : α → E
      inst✝² : One F
      inst✝¹ : NormOneClass F
      inst✝ : TopologicalSpace α
      c : Real
      b : α
      hc : Filter.Eventually (fun a => LE.le (Norm.norm (f a)) c) (nhdsWithin b (Has …
      hb : Membership.mem (HasCompl.compl (Singleton.singleton b)) b → LE.le (Norm.n …
      ⊢ LE.le (Norm.norm (f b)) (Max.max c (Norm.norm (f b)))
    -/
  · apply le_max_right
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      α : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝⁴ : Norm E
      inst✝³ : Norm F
      f : α → E
      inst✝² : One F
      inst✝¹ : NormOneClass F
      inst✝ : TopologicalSpace α
      a : α
      c : Real
      hc : Filter.Eventually (fun a => LE.le (Norm.norm (f a)) c) (nhdsWithin a (Has …
      b : α
      hb : Membership.mem (HasCompl.compl (Singleton.singleton a)) b → LE.le (Norm.n …
      hb' : Ne b a
      ⊢ LE.le (Norm.norm (f b)) (Max.max c (Norm.norm (f a)))
    -/
  · exact (hb hb').trans (le_max_left ..)
    /-
      🎉 no goals
    -/


theorem isLittleO_const_iff {c : F''} (hc : c ≠ 0) :
    (f'' =o[l] fun _x => c) ↔ Tendsto f'' l (𝓝 0) :=
  (isLittleO_const_iff_isLittleO_one ℝ hc).trans (isLittleO_one_iff _)


theorem isLittleO_id_const {c : F''} (hc : c ≠ 0) : (fun x : E'' => x) =o[𝓝 0] fun _x => c :=
  (isLittleO_const_iff hc).mpr (continuous_id.tendsto 0)


theorem _root_.Filter.IsBoundedUnder.isBigO_const (h : IsBoundedUnder (· ≤ ·) l (norm ∘ f))
    {c : F''} (hc : c ≠ 0) : f =O[l] fun _x => c :=
  (h.isBigO_one ℝ).trans (isBigO_const_const _ hc _)


theorem isBigO_const_of_tendsto {y : E''} (h : Tendsto f'' l (𝓝 y)) {c : F''} (hc : c ≠ 0) :
    f'' =O[l] fun _x => c :=
  h.norm.isBoundedUnder_le.isBigO_const hc


theorem IsBigO.isBoundedUnder_le {c : F} (h : f =O[l] fun _x => c) :
    IsBoundedUnder (· ≤ ·) l (norm ∘ f) :=
  let ⟨c', hc'⟩ := h.bound
  ⟨c' * ‖c‖, eventually_map.2 hc'⟩


theorem isBigO_const_of_ne {c : F''} (hc : c ≠ 0) :
    (f =O[l] fun _x => c) ↔ IsBoundedUnder (· ≤ ·) l (norm ∘ f) :=
  ⟨fun h => h.isBoundedUnder_le, fun h => h.isBigO_const hc⟩


theorem isBigO_const_iff {c : F''} : (f'' =O[l] fun _x => c) ↔
    (c = 0 → f'' =ᶠ[l] 0) ∧ IsBoundedUnder (· ≤ ·) l fun x => ‖f'' x‖ := by
  /-
    α : Type u_1
    E'' : Type u_9
    F'' : Type u_10
    inst✝¹ : NormedAddCommGroup E''
    inst✝ : NormedAddCommGroup F''
    f'' : α → E''
    l : Filter α
    c : F''
    ⊢ Iff (Asymptotics.IsBigO l f'' fun _x => c) (And (Eq c 0 → l.EventuallyEq f'' …
  -/
  refine ⟨fun h => ⟨fun hc => isBigO_zero_right_iff.1 (by rwa [← hc]), h.isBoundedUnder_le⟩, ?_⟩
  /-
    α : Type u_1
    E'' : Type u_9
    F'' : Type u_10
    inst✝¹ : NormedAddCommGroup E''
    inst✝ : NormedAddCommGroup F''
    f'' : α → E''
    l : Filter α
    c : F''
    ⊢ And (Eq c 0 → l.EventuallyEq f'' 0) (Filter.IsBoundedUnder (fun x1 x2 => LE. …
  -/
  rintro ⟨hcf, hf⟩
  /-
    case intro
    α : Type u_1
    E'' : Type u_9
    F'' : Type u_10
    inst✝¹ : NormedAddCommGroup E''
    inst✝ : NormedAddCommGroup F''
    f'' : α → E''
    l : Filter α
    c : F''
    hcf : Eq c 0 → l.EventuallyEq f'' 0
    hf : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l fun x => Norm.norm (f' …
    ⊢ Asymptotics.IsBigO l f'' fun _x => c
  -/
  rcases eq_or_ne c 0 with (hc | hc)
  /-
    case intro.inl
    α : Type u_1
    E'' : Type u_9
    F'' : Type u_10
    inst✝¹ : NormedAddCommGroup E''
    inst✝ : NormedAddCommGroup F''
    f'' : α → E''
    l : Filter α
    c : F''
    hcf : Eq c 0 → l.EventuallyEq f'' 0
    hf : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l fun x => Norm.norm (f' …
    hc : Eq c 0
    ⊢ Asymptotics.IsBigO l f'' fun _x => c
  -/
  exacts [(hcf hc).trans_isBigO (isBigO_zero _ _), hf.isBigO_const hc]
  /-
    🎉 no goals
  -/


theorem isBigO_iff_isBoundedUnder_le_div (h : ∀ᶠ x in l, g'' x ≠ 0) :
    f =O[l] g'' ↔ IsBoundedUnder (· ≤ ·) l fun x => ‖f x‖ / ‖g'' x‖ := by
  /-
    α : Type u_1
    E : Type u_3
    F'' : Type u_10
    inst✝¹ : Norm E
    inst✝ : NormedAddCommGroup F''
    f : α → E
    g'' : α → F''
    l : Filter α
    h : Filter.Eventually (fun x => Ne (g'' x) 0) l
    ⊢ Iff (Asymptotics.IsBigO l f g'') (Filter.IsBoundedUnder (fun x1 x2 => LE.le  …
  -/
  simp only [isBigO_iff, IsBoundedUnder, IsBounded, eventually_map]
  exact
    exists_congr fun c =>
      eventually_congr <| h.mono fun x hx => (div_le_iff₀ <| norm_pos_iff.2 hx).symm


/-- `(fun x ↦ c) =O[l] f` if and only if `f` is bounded away from zero. -/
theorem isBigO_const_left_iff_pos_le_norm {c : E''} (hc : c ≠ 0) :
    (fun _x => c) =O[l] f' ↔ ∃ b, 0 < b ∧ ∀ᶠ x in l, b ≤ ‖f' x‖ := by
  /-
    α : Type u_1
    E' : Type u_6
    E'' : Type u_9
    inst✝¹ : SeminormedAddCommGroup E'
    inst✝ : NormedAddCommGroup E''
    f' : α → E'
    l : Filter α
    c : E''
    hc : Ne c 0
    ⊢ Iff (Asymptotics.IsBigO l (fun _x => c) f') (Exists fun b => And (LT.lt 0 b) …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      E' : Type u_6
      E'' : Type u_9
      inst✝¹ : SeminormedAddCommGroup E'
      inst✝ : NormedAddCommGroup E''
      f' : α → E'
      l : Filter α
      c : E''
      hc : Ne c 0
      ⊢ Asymptotics.IsBigO l (fun _x => c) f' → Exists fun b => And (LT.lt 0 b) (Fil …
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      E' : Type u_6
      E'' : Type u_9
      inst✝¹ : SeminormedAddCommGroup E'
      inst✝ : NormedAddCommGroup E''
      f' : α → E'
      l : Filter α
      c : E''
      hc : Ne c 0
      h : Asymptotics.IsBigO l (fun _x => c) f'
      ⊢ Exists fun b => And (LT.lt 0 b) (Filter.Eventually (fun x => LE.le b (Norm.n …
    -/
    rcases h.exists_pos with ⟨C, hC₀, hC⟩
    /-
      case mp.intro.intro
      α : Type u_1
      E' : Type u_6
      E'' : Type u_9
      inst✝¹ : SeminormedAddCommGroup E'
      inst✝ : NormedAddCommGroup E''
      f' : α → E'
      l : Filter α
      c : E''
      hc : Ne c 0
      h : Asymptotics.IsBigO l (fun _x => c) f'
      C : Real
      hC₀ : GT.gt C 0
      hC : Asymptotics.IsBigOWith C l (fun _x => c) f'
      ⊢ Exists fun b => And (LT.lt 0 b) (Filter.Eventually (fun x => LE.le b (Norm.n …
    -/
    refine ⟨‖c‖ / C, div_pos (norm_pos_iff.2 hc) hC₀, ?_⟩
    /-
      case mp.intro.intro
      α : Type u_1
      E' : Type u_6
      E'' : Type u_9
      inst✝¹ : SeminormedAddCommGroup E'
      inst✝ : NormedAddCommGroup E''
      f' : α → E'
      l : Filter α
      c : E''
      hc : Ne c 0
      h : Asymptotics.IsBigO l (fun _x => c) f'
      C : Real
      hC₀ : GT.gt C 0
      hC : Asymptotics.IsBigOWith C l (fun _x => c) f'
      ⊢ Filter.Eventually (fun x => LE.le (HDiv.hDiv (Norm.norm c) C) (Norm.norm (f' …
    -/
    exact hC.bound.mono fun x => (div_le_iff₀' hC₀).2
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      E' : Type u_6
      E'' : Type u_9
      inst✝¹ : SeminormedAddCommGroup E'
      inst✝ : NormedAddCommGroup E''
      f' : α → E'
      l : Filter α
      c : E''
      hc : Ne c 0
      ⊢ (Exists fun b => And (LT.lt 0 b) (Filter.Eventually (fun x => LE.le b (Norm. …
    -/
  · rintro ⟨b, hb₀, hb⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      E' : Type u_6
      E'' : Type u_9
      inst✝¹ : SeminormedAddCommGroup E'
      inst✝ : NormedAddCommGroup E''
      f' : α → E'
      l : Filter α
      c : E''
      hc : Ne c 0
      b : Real
      hb₀ : LT.lt 0 b
      hb : Filter.Eventually (fun x => LE.le b (Norm.norm (f' x))) l
      ⊢ Asymptotics.IsBigO l (fun _x => c) f'
    -/
    refine IsBigO.of_bound (‖c‖ / b) (hb.mono fun x hx => ?_)
    /-
      case mpr.intro.intro
      α : Type u_1
      E' : Type u_6
      E'' : Type u_9
      inst✝¹ : SeminormedAddCommGroup E'
      inst✝ : NormedAddCommGroup E''
      f' : α → E'
      l : Filter α
      c : E''
      hc : Ne c 0
      b : Real
      hb₀ : LT.lt 0 b
      hb : Filter.Eventually (fun x => LE.le b (Norm.norm (f' x))) l
      x : α
      hx : LE.le b (Norm.norm (f' x))
      ⊢ LE.le (Norm.norm c) (HMul.hMul (HDiv.hDiv (Norm.norm c) b) (Norm.norm (f' x)))
    -/
    rw [div_mul_eq_mul_div, mul_div_assoc]
    /-
      case mpr.intro.intro
      α : Type u_1
      E' : Type u_6
      E'' : Type u_9
      inst✝¹ : SeminormedAddCommGroup E'
      inst✝ : NormedAddCommGroup E''
      f' : α → E'
      l : Filter α
      c : E''
      hc : Ne c 0
      b : Real
      hb₀ : LT.lt 0 b
      hb : Filter.Eventually (fun x => LE.le b (Norm.norm (f' x))) l
      x : α
      hx : LE.le b (Norm.norm (f' x))
      ⊢ LE.le (Norm.norm c) (HMul.hMul (Norm.norm c) (HDiv.hDiv (Norm.norm (f' x)) b))
    -/
    exact le_mul_of_one_le_right (norm_nonneg _) ((one_le_div hb₀).2 hx)
    /-
      🎉 no goals
    -/


theorem IsBigO.trans_tendsto (hfg : f'' =O[l] g'') (hg : Tendsto g'' l (𝓝 0)) :
    Tendsto f'' l (𝓝 0) :=
  (isLittleO_one_iff ℝ).1 <| hfg.trans_isLittleO <| (isLittleO_one_iff ℝ).2 hg


theorem IsLittleO.trans_tendsto (hfg : f'' =o[l] g'') (hg : Tendsto g'' l (𝓝 0)) :
    Tendsto f'' l (𝓝 0) :=
  hfg.isBigO.trans_tendsto hg


lemma isLittleO_id_one [One F''] [NeZero (1 : F'')] : (fun x : E'' => x) =o[𝓝 0] (1 : E'' → F'') :=
  isLittleO_id_const one_ne_zero


theorem continuousAt_iff_isLittleO {α : Type*} {E : Type*} [NormedRing E] [NormOneClass E]
    [TopologicalSpace α] {f : α → E} {x : α} :
    (ContinuousAt f x) ↔ (fun (y : α) ↦ f y - f x) =o[𝓝 x] (fun (_ : α) ↦ (1 : E)) := by
  /-
    α : Type u_17
    E : Type u_18
    inst✝² : NormedRing E
    inst✝¹ : NormOneClass E
    inst✝ : TopologicalSpace α
    f : α → E
    x : α
    ⊢ Iff (ContinuousAt f x) (Asymptotics.IsLittleO (nhds x) (fun y => HSub.hSub ( …
  -/
  simp [ContinuousAt, ← tendsto_sub_nhds_zero_iff]
  /-
    🎉 no goals
  -/


theorem isBigOWith_const_mul_self (c : R) (f : α → R) (l : Filter α) :
    IsBigOWith ‖c‖ l (fun x => c * f x) f :=
  isBigOWith_of_le' _ fun _x => norm_mul_le _ _


theorem isBigO_const_mul_self (c : R) (f : α → R) (l : Filter α) : (fun x => c * f x) =O[l] f :=
  (isBigOWith_const_mul_self c f l).isBigO


theorem IsBigOWith.const_mul_left {f : α → R} (h : IsBigOWith c l f g) (c' : R) :
    IsBigOWith (‖c'‖ * c) l (fun x => c' * f x) g :=
  (isBigOWith_const_mul_self c' f l).trans h (norm_nonneg c')


theorem IsBigO.const_mul_left {f : α → R} (h : f =O[l] g) (c' : R) : (fun x => c' * f x) =O[l] g :=
  let ⟨_c, hc⟩ := h.isBigOWith
  (hc.const_mul_left c').isBigO


theorem isBigOWith_self_const_mul' (u : Rˣ) (f : α → R) (l : Filter α) :
    IsBigOWith ‖(↑u⁻¹ : R)‖ l f fun x => ↑u * f x :=
  (isBigOWith_const_mul_self ↑u⁻¹ (fun x ↦ ↑u * f x) l).congr_left
    fun x ↦ u.inv_mul_cancel_left (f x)


theorem isBigOWith_self_const_mul (c : 𝕜) (hc : c ≠ 0) (f : α → 𝕜) (l : Filter α) :
    IsBigOWith ‖c‖⁻¹ l f fun x => c * f x :=
  (isBigOWith_self_const_mul' (Units.mk0 c hc) f l).congr_const <| norm_inv c


theorem isBigO_self_const_mul' {c : R} (hc : IsUnit c) (f : α → R) (l : Filter α) :
    f =O[l] fun x => c * f x :=
  let ⟨u, hu⟩ := hc
  hu ▸ (isBigOWith_self_const_mul' u f l).isBigO


theorem isBigO_self_const_mul (c : 𝕜) (hc : c ≠ 0) (f : α → 𝕜) (l : Filter α) :
    f =O[l] fun x => c * f x :=
  isBigO_self_const_mul' (IsUnit.mk0 c hc) f l


theorem isBigO_const_mul_left_iff' {f : α → R} {c : R} (hc : IsUnit c) :
    (fun x => c * f x) =O[l] g ↔ f =O[l] g :=
  ⟨(isBigO_self_const_mul' hc f l).trans, fun h => h.const_mul_left c⟩


theorem isBigO_const_mul_left_iff {f : α → 𝕜} {c : 𝕜} (hc : c ≠ 0) :
    (fun x => c * f x) =O[l] g ↔ f =O[l] g :=
  isBigO_const_mul_left_iff' <| IsUnit.mk0 c hc


theorem IsLittleO.const_mul_left {f : α → R} (h : f =o[l] g) (c : R) : (fun x => c * f x) =o[l] g :=
  (isBigO_const_mul_self c f l).trans_isLittleO h


theorem isLittleO_const_mul_left_iff' {f : α → R} {c : R} (hc : IsUnit c) :
    (fun x => c * f x) =o[l] g ↔ f =o[l] g :=
  ⟨(isBigO_self_const_mul' hc f l).trans_isLittleO, fun h => h.const_mul_left c⟩


theorem isLittleO_const_mul_left_iff {f : α → 𝕜} {c : 𝕜} (hc : c ≠ 0) :
    (fun x => c * f x) =o[l] g ↔ f =o[l] g :=
  isLittleO_const_mul_left_iff' <| IsUnit.mk0 c hc


theorem IsBigOWith.of_const_mul_right {g : α → R} {c : R} (hc' : 0 ≤ c')
    (h : IsBigOWith c' l f fun x => c * g x) : IsBigOWith (c' * ‖c‖) l f g :=
  h.trans (isBigOWith_const_mul_self c g l) hc'


theorem IsBigO.of_const_mul_right {g : α → R} {c : R} (h : f =O[l] fun x => c * g x) : f =O[l] g :=
  let ⟨_c, cnonneg, hc⟩ := h.exists_nonneg
  (hc.of_const_mul_right cnonneg).isBigO


theorem IsBigOWith.const_mul_right' {g : α → R} {u : Rˣ} {c' : ℝ} (hc' : 0 ≤ c')
    (h : IsBigOWith c' l f g) : IsBigOWith (c' * ‖(↑u⁻¹ : R)‖) l f fun x => ↑u * g x :=
  h.trans (isBigOWith_self_const_mul' _ _ _) hc'


theorem IsBigOWith.const_mul_right {g : α → 𝕜} {c : 𝕜} (hc : c ≠ 0) {c' : ℝ} (hc' : 0 ≤ c')
    (h : IsBigOWith c' l f g) : IsBigOWith (c' * ‖c‖⁻¹) l f fun x => c * g x :=
  h.trans (isBigOWith_self_const_mul c hc g l) hc'


theorem IsBigO.const_mul_right' {g : α → R} {c : R} (hc : IsUnit c) (h : f =O[l] g) :
    f =O[l] fun x => c * g x :=
  h.trans (isBigO_self_const_mul' hc g l)


theorem IsBigO.const_mul_right {g : α → 𝕜} {c : 𝕜} (hc : c ≠ 0) (h : f =O[l] g) :
    f =O[l] fun x => c * g x :=
  h.const_mul_right' <| IsUnit.mk0 c hc


theorem isBigO_const_mul_right_iff' {g : α → R} {c : R} (hc : IsUnit c) :
    (f =O[l] fun x => c * g x) ↔ f =O[l] g :=
  ⟨fun h => h.of_const_mul_right, fun h => h.const_mul_right' hc⟩


theorem isBigO_const_mul_right_iff {g : α → 𝕜} {c : 𝕜} (hc : c ≠ 0) :
    (f =O[l] fun x => c * g x) ↔ f =O[l] g :=
  isBigO_const_mul_right_iff' <| IsUnit.mk0 c hc


theorem IsLittleO.of_const_mul_right {g : α → R} {c : R} (h : f =o[l] fun x => c * g x) :
    f =o[l] g :=
  h.trans_isBigO (isBigO_const_mul_self c g l)


theorem IsLittleO.const_mul_right' {g : α → R} {c : R} (hc : IsUnit c) (h : f =o[l] g) :
    f =o[l] fun x => c * g x :=
  h.trans_isBigO (isBigO_self_const_mul' hc g l)


theorem IsLittleO.const_mul_right {g : α → 𝕜} {c : 𝕜} (hc : c ≠ 0) (h : f =o[l] g) :
    f =o[l] fun x => c * g x :=
  h.const_mul_right' <| IsUnit.mk0 c hc


theorem isLittleO_const_mul_right_iff' {g : α → R} {c : R} (hc : IsUnit c) :
    (f =o[l] fun x => c * g x) ↔ f =o[l] g :=
  ⟨fun h => h.of_const_mul_right, fun h => h.const_mul_right' hc⟩


theorem isLittleO_const_mul_right_iff {g : α → 𝕜} {c : 𝕜} (hc : c ≠ 0) :
    (f =o[l] fun x => c * g x) ↔ f =o[l] g :=
  isLittleO_const_mul_right_iff' <| IsUnit.mk0 c hc


theorem IsBigOWith.mul {f₁ f₂ : α → R} {g₁ g₂ : α → 𝕜} {c₁ c₂ : ℝ} (h₁ : IsBigOWith c₁ l f₁ g₁)
    (h₂ : IsBigOWith c₂ l f₂ g₂) :
    IsBigOWith (c₁ * c₂) l (fun x => f₁ x * f₂ x) fun x => g₁ x * g₂ x := by
  /-
    α : Type u_1
    R : Type u_13
    𝕜 : Type u_15
    inst✝¹ : SeminormedRing R
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    f₁ f₂ : α → R
    g₁ g₂ : α → 𝕜
    c₁ c₂ : Real
    h₁ : Asymptotics.IsBigOWith c₁ l f₁ g₁
    h₂ : Asymptotics.IsBigOWith c₂ l f₂ g₂
    ⊢ Asymptotics.IsBigOWith (HMul.hMul c₁ c₂) l (fun x => HMul.hMul (f₁ x) (f₂ x) …
  -/
  simp only [IsBigOWith_def] at *
  /-
    α : Type u_1
    R : Type u_13
    𝕜 : Type u_15
    inst✝¹ : SeminormedRing R
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    f₁ f₂ : α → R
    g₁ g₂ : α → 𝕜
    c₁ c₂ : Real
    h₁ : Filter.Eventually (fun x => LE.le (Norm.norm (f₁ x)) (HMul.hMul c₁ (Norm. …
    h₂ : Filter.Eventually (fun x => LE.le (Norm.norm (f₂ x)) (HMul.hMul c₂ (Norm. …
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HMul.hMul (f₁ x) (f₂ x))) (HMu …
  -/
  filter_upwards [h₁, h₂] with _ hx₁ hx₂
  /-
    case h
    α : Type u_1
    R : Type u_13
    𝕜 : Type u_15
    inst✝¹ : SeminormedRing R
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    f₁ f₂ : α → R
    g₁ g₂ : α → 𝕜
    c₁ c₂ : Real
    h₁ : Filter.Eventually (fun x => LE.le (Norm.norm (f₁ x)) (HMul.hMul c₁ (Norm. …
    h₂ : Filter.Eventually (fun x => LE.le (Norm.norm (f₂ x)) (HMul.hMul c₂ (Norm. …
    a✝ : α
    hx₁ : LE.le (Norm.norm (f₁ a✝)) (HMul.hMul c₁ (Norm.norm (g₁ a✝)))
    hx₂ : LE.le (Norm.norm (f₂ a✝)) (HMul.hMul c₂ (Norm.norm (g₂ a✝)))
    ⊢ LE.le (Norm.norm (HMul.hMul (f₁ a✝) (f₂ a✝))) (HMul.hMul (HMul.hMul c₁ c₂) ( …
  -/
  apply le_trans (norm_mul_le _ _)
  /-
    case h
    α : Type u_1
    R : Type u_13
    𝕜 : Type u_15
    inst✝¹ : SeminormedRing R
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    f₁ f₂ : α → R
    g₁ g₂ : α → 𝕜
    c₁ c₂ : Real
    h₁ : Filter.Eventually (fun x => LE.le (Norm.norm (f₁ x)) (HMul.hMul c₁ (Norm. …
    h₂ : Filter.Eventually (fun x => LE.le (Norm.norm (f₂ x)) (HMul.hMul c₂ (Norm. …
    a✝ : α
    hx₁ : LE.le (Norm.norm (f₁ a✝)) (HMul.hMul c₁ (Norm.norm (g₁ a✝)))
    hx₂ : LE.le (Norm.norm (f₂ a✝)) (HMul.hMul c₂ (Norm.norm (g₂ a✝)))
    ⊢ LE.le (HMul.hMul (Norm.norm (f₁ a✝)) (Norm.norm (f₂ a✝))) (HMul.hMul (HMul.h …
  -/
  convert mul_le_mul hx₁ hx₂ (norm_nonneg _) (le_trans (norm_nonneg _) hx₁) using 1
  /-
    case h.e'_4
    α : Type u_1
    R : Type u_13
    𝕜 : Type u_15
    inst✝¹ : SeminormedRing R
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    f₁ f₂ : α → R
    g₁ g₂ : α → 𝕜
    c₁ c₂ : Real
    h₁ : Filter.Eventually (fun x => LE.le (Norm.norm (f₁ x)) (HMul.hMul c₁ (Norm. …
    h₂ : Filter.Eventually (fun x => LE.le (Norm.norm (f₂ x)) (HMul.hMul c₂ (Norm. …
    a✝ : α
    hx₁ : LE.le (Norm.norm (f₁ a✝)) (HMul.hMul c₁ (Norm.norm (g₁ a✝)))
    hx₂ : LE.le (Norm.norm (f₂ a✝)) (HMul.hMul c₂ (Norm.norm (g₂ a✝)))
    ⊢ Eq (HMul.hMul (HMul.hMul c₁ c₂) (Norm.norm (HMul.hMul (g₁ a✝) (g₂ a✝)))) (HM …
  -/
  rw [norm_mul, mul_mul_mul_comm]
  /-
    🎉 no goals
  -/


theorem IsBigO.mul {f₁ f₂ : α → R} {g₁ g₂ : α → 𝕜} (h₁ : f₁ =O[l] g₁) (h₂ : f₂ =O[l] g₂) :
    (fun x => f₁ x * f₂ x) =O[l] fun x => g₁ x * g₂ x :=
  let ⟨_c, hc⟩ := h₁.isBigOWith
  let ⟨_c', hc'⟩ := h₂.isBigOWith
  (hc.mul hc').isBigO


theorem IsBigO.mul_isLittleO {f₁ f₂ : α → R} {g₁ g₂ : α → 𝕜} (h₁ : f₁ =O[l] g₁) (h₂ : f₂ =o[l] g₂) :
    (fun x => f₁ x * f₂ x) =o[l] fun x => g₁ x * g₂ x := by
  /-
    α : Type u_1
    R : Type u_13
    𝕜 : Type u_15
    inst✝¹ : SeminormedRing R
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    f₁ f₂ : α → R
    g₁ g₂ : α → 𝕜
    h₁ : Asymptotics.IsBigO l f₁ g₁
    h₂ : Asymptotics.IsLittleO l f₂ g₂
    ⊢ Asymptotics.IsLittleO l (fun x => HMul.hMul (f₁ x) (f₂ x)) fun x => HMul.hMu …
  -/
  simp only [IsLittleO_def] at *
  /-
    α : Type u_1
    R : Type u_13
    𝕜 : Type u_15
    inst✝¹ : SeminormedRing R
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    f₁ f₂ : α → R
    g₁ g₂ : α → 𝕜
    h₁ : Asymptotics.IsBigO l f₁ g₁
    h₂ : ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l f₂ g₂
    ⊢ ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l (fun x => HMul.hMul (f₁ …
  -/
  intro c cpos
  /-
    α : Type u_1
    R : Type u_13
    𝕜 : Type u_15
    inst✝¹ : SeminormedRing R
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    f₁ f₂ : α → R
    g₁ g₂ : α → 𝕜
    h₁ : Asymptotics.IsBigO l f₁ g₁
    h₂ : ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l f₂ g₂
    c : Real
    cpos : LT.lt 0 c
    ⊢ Asymptotics.IsBigOWith c l (fun x => HMul.hMul (f₁ x) (f₂ x)) fun x => HMul. …
  -/
  rcases h₁.exists_pos with ⟨c', c'pos, hc'⟩
  /-
    case intro.intro
    α : Type u_1
    R : Type u_13
    𝕜 : Type u_15
    inst✝¹ : SeminormedRing R
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    f₁ f₂ : α → R
    g₁ g₂ : α → 𝕜
    h₁ : Asymptotics.IsBigO l f₁ g₁
    h₂ : ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l f₂ g₂
    c : Real
    cpos : LT.lt 0 c
    c' : Real
    c'pos : GT.gt c' 0
    hc' : Asymptotics.IsBigOWith c' l f₁ g₁
    ⊢ Asymptotics.IsBigOWith c l (fun x => HMul.hMul (f₁ x) (f₂ x)) fun x => HMul. …
  -/
  exact (hc'.mul (h₂ (div_pos cpos c'pos))).congr_const (mul_div_cancel₀ _ (ne_of_gt c'pos))
  /-
    🎉 no goals
  -/


theorem IsLittleO.mul_isBigO {f₁ f₂ : α → R} {g₁ g₂ : α → 𝕜} (h₁ : f₁ =o[l] g₁) (h₂ : f₂ =O[l] g₂) :
    (fun x => f₁ x * f₂ x) =o[l] fun x => g₁ x * g₂ x := by
  /-
    α : Type u_1
    R : Type u_13
    𝕜 : Type u_15
    inst✝¹ : SeminormedRing R
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    f₁ f₂ : α → R
    g₁ g₂ : α → 𝕜
    h₁ : Asymptotics.IsLittleO l f₁ g₁
    h₂ : Asymptotics.IsBigO l f₂ g₂
    ⊢ Asymptotics.IsLittleO l (fun x => HMul.hMul (f₁ x) (f₂ x)) fun x => HMul.hMu …
  -/
  simp only [IsLittleO_def] at *
  /-
    α : Type u_1
    R : Type u_13
    𝕜 : Type u_15
    inst✝¹ : SeminormedRing R
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    f₁ f₂ : α → R
    g₁ g₂ : α → 𝕜
    h₂ : Asymptotics.IsBigO l f₂ g₂
    h₁ : ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l f₁ g₁
    ⊢ ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l (fun x => HMul.hMul (f₁ …
  -/
  intro c cpos
  /-
    α : Type u_1
    R : Type u_13
    𝕜 : Type u_15
    inst✝¹ : SeminormedRing R
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    f₁ f₂ : α → R
    g₁ g₂ : α → 𝕜
    h₂ : Asymptotics.IsBigO l f₂ g₂
    h₁ : ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l f₁ g₁
    c : Real
    cpos : LT.lt 0 c
    ⊢ Asymptotics.IsBigOWith c l (fun x => HMul.hMul (f₁ x) (f₂ x)) fun x => HMul. …
  -/
  rcases h₂.exists_pos with ⟨c', c'pos, hc'⟩
  /-
    case intro.intro
    α : Type u_1
    R : Type u_13
    𝕜 : Type u_15
    inst✝¹ : SeminormedRing R
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    f₁ f₂ : α → R
    g₁ g₂ : α → 𝕜
    h₂ : Asymptotics.IsBigO l f₂ g₂
    h₁ : ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l f₁ g₁
    c : Real
    cpos : LT.lt 0 c
    c' : Real
    c'pos : GT.gt c' 0
    hc' : Asymptotics.IsBigOWith c' l f₂ g₂
    ⊢ Asymptotics.IsBigOWith c l (fun x => HMul.hMul (f₁ x) (f₂ x)) fun x => HMul. …
  -/
  exact ((h₁ (div_pos cpos c'pos)).mul hc').congr_const (div_mul_cancel₀ _ (ne_of_gt c'pos))
  /-
    🎉 no goals
  -/


theorem IsLittleO.mul {f₁ f₂ : α → R} {g₁ g₂ : α → 𝕜} (h₁ : f₁ =o[l] g₁) (h₂ : f₂ =o[l] g₂) :
    (fun x => f₁ x * f₂ x) =o[l] fun x => g₁ x * g₂ x :=
  h₁.mul_isBigO h₂.isBigO


theorem IsBigOWith.pow' {f : α → R} {g : α → 𝕜} (h : IsBigOWith c l f g) :
    ∀ n : ℕ, IsBigOWith (Nat.casesOn n ‖(1 : R)‖ fun n => c ^ (n + 1))
      l (fun x => f x ^ n) fun x => g x ^ n
            /-
              α : Type u_1
              R : Type u_13
              𝕜 : Type u_15
              inst✝¹ : SeminormedRing R
              inst✝ : NormedDivisionRing 𝕜
              c : Real
              l : Filter α
              f : α → R
              g : α → 𝕜
              h : Asymptotics.IsBigOWith c l f g
              ⊢ Asymptotics.IsBigOWith (Nat.casesOn 0 (Norm.norm 1) fun n => HPow.hPow c (HA …
            -/
  | 0 => by simpa using isBigOWith_const_const (1 : R) (one_ne_zero' 𝕜) l
            /-
              🎉 no goals
            -/
            /-
              α : Type u_1
              R : Type u_13
              𝕜 : Type u_15
              inst✝¹ : SeminormedRing R
              inst✝ : NormedDivisionRing 𝕜
              c : Real
              l : Filter α
              f : α → R
              g : α → 𝕜
              h : Asymptotics.IsBigOWith c l f g
              ⊢ Asymptotics.IsBigOWith (Nat.casesOn 1 (Norm.norm 1) fun n => HPow.hPow c (HA …
            -/
  | 1 => by simpa
            /-
              🎉 no goals
            -/
                /-
                  α : Type u_1
                  R : Type u_13
                  𝕜 : Type u_15
                  inst✝¹ : SeminormedRing R
                  inst✝ : NormedDivisionRing 𝕜
                  c : Real
                  l : Filter α
                  f : α → R
                  g : α → 𝕜
                  h : Asymptotics.IsBigOWith c l f g
                  n : Nat
                  ⊢ Asymptotics.IsBigOWith (Nat.casesOn (HAdd.hAdd n 2) (Norm.norm 1) fun n => H …
                -/
  | n + 2 => by simpa [pow_succ] using (IsBigOWith.pow' h (n + 1)).mul h
                /-
                  🎉 no goals
                -/


theorem IsBigOWith.pow [NormOneClass R] {f : α → R} {g : α → 𝕜} (h : IsBigOWith c l f g) :
    ∀ n : ℕ, IsBigOWith (c ^ n) l (fun x => f x ^ n) fun x => g x ^ n
            /-
              α : Type u_1
              R : Type u_13
              𝕜 : Type u_15
              inst✝² : SeminormedRing R
              inst✝¹ : NormedDivisionRing 𝕜
              c : Real
              l : Filter α
              inst✝ : NormOneClass R
              f : α → R
              g : α → 𝕜
              h : Asymptotics.IsBigOWith c l f g
              ⊢ Asymptotics.IsBigOWith (HPow.hPow c 0) l (fun x => HPow.hPow (f x) 0) fun x  …
            -/
  | 0 => by simpa using h.pow' 0
            /-
              🎉 no goals
            -/
  | n + 1 => h.pow' (n + 1)


theorem IsBigOWith.of_pow {n : ℕ} {f : α → 𝕜} {g : α → R} (h : IsBigOWith c l (f ^ n) (g ^ n))
    (hn : n ≠ 0) (hc : c ≤ c' ^ n) (hc' : 0 ≤ c') : IsBigOWith c' l f g :=
  IsBigOWith.of_bound <| (h.weaken hc).bound.mono fun x hx ↦
                                  /-
                                    α : Type u_1
                                    R : Type u_13
                                    𝕜 : Type u_15
                                    inst✝¹ : SeminormedRing R
                                    inst✝ : NormedDivisionRing 𝕜
                                    c c' : Real
                                    l : Filter α
                                    n : Nat
                                    f : α → 𝕜
                                    g : α → R
                                    h : Asymptotics.IsBigOWith c l (HPow.hPow f n) (HPow.hPow g n)
                                    hn : Ne n 0
                                    hc : LE.le c (HPow.hPow c' n)
                                    hc' : LE.le 0 c'
                                    x : α
                                    hx : LE.le (Norm.norm (HPow.hPow f n x)) (HMul.hMul (HPow.hPow c' n) (Norm.nor …
                                    ⊢ LE.le 0 (HMul.hMul c' (Norm.norm (g x)))
                                  -/
    le_of_pow_le_pow_left₀ hn (by positivity) <|
                                  /-
                                    🎉 no goals
                                  -/
      calc
        ‖f x‖ ^ n = ‖f x ^ n‖ := (norm_pow _ _).symm
        _ ≤ c' ^ n * ‖g x ^ n‖ := hx
                                     /-
                                       α : Type u_1
                                       R : Type u_13
                                       𝕜 : Type u_15
                                       inst✝¹ : SeminormedRing R
                                       inst✝ : NormedDivisionRing 𝕜
                                       c c' : Real
                                       l : Filter α
                                       n : Nat
                                       f : α → 𝕜
                                       g : α → R
                                       h : Asymptotics.IsBigOWith c l (HPow.hPow f n) (HPow.hPow g n)
                                       hn : Ne n 0
                                       hc : LE.le c (HPow.hPow c' n)
                                       hc' : LE.le 0 c'
                                       x : α
                                       hx : LE.le (Norm.norm (HPow.hPow f n x)) (HMul.hMul (HPow.hPow c' n) (Norm.nor …
                                       ⊢ LE.le (HMul.hMul (HPow.hPow c' n) (Norm.norm (HPow.hPow (g x) n))) (HMul.hMu …
                                     -/
        _ ≤ c' ^ n * ‖g x‖ ^ n := by gcongr; exact norm_pow_le' _ hn.bot_lt
                                             /-
                                               🎉 no goals
                                             -/
        _ = (c' * ‖g x‖) ^ n := (mul_pow _ _ _).symm


theorem IsBigO.pow {f : α → R} {g : α → 𝕜} (h : f =O[l] g) (n : ℕ) :
    (fun x => f x ^ n) =O[l] fun x => g x ^ n :=
  let ⟨_C, hC⟩ := h.isBigOWith
  isBigO_iff_isBigOWith.2 ⟨_, hC.pow' n⟩


theorem IsBigO.of_pow {f : α → 𝕜} {g : α → R} {n : ℕ} (hn : n ≠ 0) (h : (f ^ n) =O[l] (g ^ n)) :
    f =O[l] g := by
  /-
    α : Type u_1
    R : Type u_13
    𝕜 : Type u_15
    inst✝¹ : SeminormedRing R
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    f : α → 𝕜
    g : α → R
    n : Nat
    hn : Ne n 0
    h : Asymptotics.IsBigO l (HPow.hPow f n) (HPow.hPow g n)
    ⊢ Asymptotics.IsBigO l f g
  -/
  rcases h.exists_pos with ⟨C, _hC₀, hC⟩
  obtain ⟨c : ℝ, hc₀ : 0 ≤ c, hc : C ≤ c ^ n⟩ :=
    ((eventually_ge_atTop _).and <| (tendsto_pow_atTop hn).eventually_ge_atTop C).exists
  /-
    case intro.intro.intro.intro
    α : Type u_1
    R : Type u_13
    𝕜 : Type u_15
    inst✝¹ : SeminormedRing R
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    f : α → 𝕜
    g : α → R
    n : Nat
    hn : Ne n 0
    h : Asymptotics.IsBigO l (HPow.hPow f n) (HPow.hPow g n)
    C : Real
    _hC₀ : GT.gt C 0
    hC : Asymptotics.IsBigOWith C l (HPow.hPow f n) (HPow.hPow g n)
    c : Real
    hc₀ : LE.le 0 c
    hc : LE.le C (HPow.hPow c n)
    ⊢ Asymptotics.IsBigO l f g
  -/
  exact (hC.of_pow hn hc hc₀).isBigO
  /-
    🎉 no goals
  -/


theorem IsLittleO.pow {f : α → R} {g : α → 𝕜} (h : f =o[l] g) {n : ℕ} (hn : 0 < n) :
    (fun x => f x ^ n) =o[l] fun x => g x ^ n := by
  /-
    α : Type u_1
    R : Type u_13
    𝕜 : Type u_15
    inst✝¹ : SeminormedRing R
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    f : α → R
    g : α → 𝕜
    h : Asymptotics.IsLittleO l f g
    n : Nat
    hn : LT.lt 0 n
    ⊢ Asymptotics.IsLittleO l (fun x => HPow.hPow (f x) n) fun x => HPow.hPow (g x …
  -/
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hn.ne'; clear hn
  induction n with
  | zero => simpa only [pow_one]
  | succ n ihn => convert ihn.mul h <;> simp [pow_succ]


theorem IsLittleO.of_pow {f : α → 𝕜} {g : α → R} {n : ℕ} (h : (f ^ n) =o[l] (g ^ n)) (hn : n ≠ 0) :
    f =o[l] g :=
  IsLittleO.of_isBigOWith fun _c hc => (h.def' <| pow_pos hc _).of_pow hn le_rfl hc.le


theorem IsBigOWith.inv_rev {f : α → 𝕜} {g : α → 𝕜'} (h : IsBigOWith c l f g)
    (h₀ : ∀ᶠ x in l, f x = 0 → g x = 0) : IsBigOWith c l (fun x => (g x)⁻¹) fun x => (f x)⁻¹ := by
  /-
    α : Type u_1
    𝕜 : Type u_15
    𝕜' : Type u_16
    inst✝¹ : NormedDivisionRing 𝕜
    inst✝ : NormedDivisionRing 𝕜'
    c : Real
    l : Filter α
    f : α → 𝕜
    g : α → 𝕜'
    h : Asymptotics.IsBigOWith c l f g
    h₀ : Filter.Eventually (fun x => Eq (f x) 0 → Eq (g x) 0) l
    ⊢ Asymptotics.IsBigOWith c l (fun x => Inv.inv (g x)) fun x => Inv.inv (f x)
  -/
  refine IsBigOWith.of_bound (h.bound.mp (h₀.mono fun x h₀ hle => ?_))
  /-
    α : Type u_1
    𝕜 : Type u_15
    𝕜' : Type u_16
    inst✝¹ : NormedDivisionRing 𝕜
    inst✝ : NormedDivisionRing 𝕜'
    c : Real
    l : Filter α
    f : α → 𝕜
    g : α → 𝕜'
    h : Asymptotics.IsBigOWith c l f g
    h₀✝ : Filter.Eventually (fun x => Eq (f x) 0 → Eq (g x) 0) l
    x : α
    h₀ : Eq (f x) 0 → Eq (g x) 0
    hle : LE.le (Norm.norm (f x)) (HMul.hMul c (Norm.norm (g x)))
    ⊢ LE.le (Norm.norm (Inv.inv (g x))) (HMul.hMul c (Norm.norm (Inv.inv (f x))))
  -/
  rcases eq_or_ne (f x) 0 with hx | hx
    /-
      case inl
      α : Type u_1
      𝕜 : Type u_15
      𝕜' : Type u_16
      inst✝¹ : NormedDivisionRing 𝕜
      inst✝ : NormedDivisionRing 𝕜'
      c : Real
      l : Filter α
      f : α → 𝕜
      g : α → 𝕜'
      h : Asymptotics.IsBigOWith c l f g
      h₀✝ : Filter.Eventually (fun x => Eq (f x) 0 → Eq (g x) 0) l
      x : α
      h₀ : Eq (f x) 0 → Eq (g x) 0
      hle : LE.le (Norm.norm (f x)) (HMul.hMul c (Norm.norm (g x)))
      hx : Eq (f x) 0
      ⊢ LE.le (Norm.norm (Inv.inv (g x))) (HMul.hMul c (Norm.norm (Inv.inv (f x))))
    -/
  · simp only [hx, h₀ hx, inv_zero, norm_zero, mul_zero, le_rfl]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      𝕜 : Type u_15
      𝕜' : Type u_16
      inst✝¹ : NormedDivisionRing 𝕜
      inst✝ : NormedDivisionRing 𝕜'
      c : Real
      l : Filter α
      f : α → 𝕜
      g : α → 𝕜'
      h : Asymptotics.IsBigOWith c l f g
      h₀✝ : Filter.Eventually (fun x => Eq (f x) 0 → Eq (g x) 0) l
      x : α
      h₀ : Eq (f x) 0 → Eq (g x) 0
      hle : LE.le (Norm.norm (f x)) (HMul.hMul c (Norm.norm (g x)))
      hx : Ne (f x) 0
      ⊢ LE.le (Norm.norm (Inv.inv (g x))) (HMul.hMul c (Norm.norm (Inv.inv (f x))))
    -/
  · have hc : 0 < c := pos_of_mul_pos_left ((norm_pos_iff.2 hx).trans_le hle) (norm_nonneg _)
    /-
      case inr
      α : Type u_1
      𝕜 : Type u_15
      𝕜' : Type u_16
      inst✝¹ : NormedDivisionRing 𝕜
      inst✝ : NormedDivisionRing 𝕜'
      c : Real
      l : Filter α
      f : α → 𝕜
      g : α → 𝕜'
      h : Asymptotics.IsBigOWith c l f g
      h₀✝ : Filter.Eventually (fun x => Eq (f x) 0 → Eq (g x) 0) l
      x : α
      h₀ : Eq (f x) 0 → Eq (g x) 0
      hle : LE.le (Norm.norm (f x)) (HMul.hMul c (Norm.norm (g x)))
      hx : Ne (f x) 0
      hc : LT.lt 0 c
      ⊢ LE.le (Norm.norm (Inv.inv (g x))) (HMul.hMul c (Norm.norm (Inv.inv (f x))))
    -/
    replace hle := inv_anti₀ (norm_pos_iff.2 hx) hle
    /-
      case inr
      α : Type u_1
      𝕜 : Type u_15
      𝕜' : Type u_16
      inst✝¹ : NormedDivisionRing 𝕜
      inst✝ : NormedDivisionRing 𝕜'
      c : Real
      l : Filter α
      f : α → 𝕜
      g : α → 𝕜'
      h : Asymptotics.IsBigOWith c l f g
      h₀✝ : Filter.Eventually (fun x => Eq (f x) 0 → Eq (g x) 0) l
      x : α
      h₀ : Eq (f x) 0 → Eq (g x) 0
      hx : Ne (f x) 0
      hc : LT.lt 0 c
      hle : LE.le (Inv.inv (HMul.hMul c (Norm.norm (g x)))) (Inv.inv (Norm.norm (f x …
      ⊢ LE.le (Norm.norm (Inv.inv (g x))) (HMul.hMul c (Norm.norm (Inv.inv (f x))))
    -/
    simpa only [norm_inv, mul_inv, ← div_eq_inv_mul, div_le_iff₀ hc] using hle
    /-
      🎉 no goals
    -/


theorem IsBigO.inv_rev {f : α → 𝕜} {g : α → 𝕜'} (h : f =O[l] g)
    (h₀ : ∀ᶠ x in l, f x = 0 → g x = 0) : (fun x => (g x)⁻¹) =O[l] fun x => (f x)⁻¹ :=
  let ⟨_c, hc⟩ := h.isBigOWith
  (hc.inv_rev h₀).isBigO


theorem IsLittleO.inv_rev {f : α → 𝕜} {g : α → 𝕜'} (h : f =o[l] g)
    (h₀ : ∀ᶠ x in l, f x = 0 → g x = 0) : (fun x => (g x)⁻¹) =o[l] fun x => (f x)⁻¹ :=
  IsLittleO.of_isBigOWith fun _c hc => (h.def' hc).inv_rev h₀


theorem IsBigOWith.const_smul_self (c' : R) :
    IsBigOWith (‖c'‖) l (fun x => c' • f' x) f' :=
  isBigOWith_of_le' _ fun _ => norm_smul_le _ _


theorem IsBigO.const_smul_self (c' : R) : (fun x => c' • f' x) =O[l] f' :=
  (IsBigOWith.const_smul_self _).isBigO


theorem IsBigOWith.const_smul_left (h : IsBigOWith c l f' g) (c' : R) :
    IsBigOWith (‖c'‖ * c) l (fun x => c' • f' x) g :=
  .trans (.const_smul_self _) h (norm_nonneg _)


theorem IsBigO.const_smul_left (h : f' =O[l] g) (c : R) : (c • f') =O[l] g :=
  let ⟨_b, hb⟩ := h.isBigOWith
  (hb.const_smul_left _).isBigO


theorem IsLittleO.const_smul_left (h : f' =o[l] g) (c : R) : (c • f') =o[l] g :=
  (IsBigO.const_smul_self _).trans_isLittleO h


theorem isBigO_const_smul_left {c : 𝕜} (hc : c ≠ 0) : (fun x => c • f' x) =O[l] g ↔ f' =O[l] g := by
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    𝕜 : Type u_15
    inst✝⁴ : Norm F
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedDivisionRing 𝕜
    g : α → F
    f' : α → E'
    l : Filter α
    inst✝¹ : Module 𝕜 E'
    inst✝ : BoundedSMul 𝕜 E'
    c : 𝕜
    hc : Ne c 0
    ⊢ Iff (Asymptotics.IsBigO l (fun x => HSMul.hSMul c (f' x)) g) (Asymptotics.Is …
  -/
  have cne0 : ‖c‖ ≠ 0 := norm_ne_zero_iff.mpr hc
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    𝕜 : Type u_15
    inst✝⁴ : Norm F
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedDivisionRing 𝕜
    g : α → F
    f' : α → E'
    l : Filter α
    inst✝¹ : Module 𝕜 E'
    inst✝ : BoundedSMul 𝕜 E'
    c : 𝕜
    hc : Ne c 0
    cne0 : Ne (Norm.norm c) 0
    ⊢ Iff (Asymptotics.IsBigO l (fun x => HSMul.hSMul c (f' x)) g) (Asymptotics.Is …
  -/
  rw [← isBigO_norm_left]
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    𝕜 : Type u_15
    inst✝⁴ : Norm F
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedDivisionRing 𝕜
    g : α → F
    f' : α → E'
    l : Filter α
    inst✝¹ : Module 𝕜 E'
    inst✝ : BoundedSMul 𝕜 E'
    c : 𝕜
    hc : Ne c 0
    cne0 : Ne (Norm.norm c) 0
    ⊢ Iff (Asymptotics.IsBigO l (fun x => Norm.norm (HSMul.hSMul c (f' x))) g) (As …
  -/
  simp only [norm_smul]
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    𝕜 : Type u_15
    inst✝⁴ : Norm F
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedDivisionRing 𝕜
    g : α → F
    f' : α → E'
    l : Filter α
    inst✝¹ : Module 𝕜 E'
    inst✝ : BoundedSMul 𝕜 E'
    c : 𝕜
    hc : Ne c 0
    cne0 : Ne (Norm.norm c) 0
    ⊢ Iff (Asymptotics.IsBigO l (fun x => HMul.hMul (Norm.norm c) (Norm.norm (f' x …
  -/
  rw [isBigO_const_mul_left_iff cne0, isBigO_norm_left]
  /-
    🎉 no goals
  -/


theorem isLittleO_const_smul_left {c : 𝕜} (hc : c ≠ 0) :
    (fun x => c • f' x) =o[l] g ↔ f' =o[l] g := by
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    𝕜 : Type u_15
    inst✝⁴ : Norm F
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedDivisionRing 𝕜
    g : α → F
    f' : α → E'
    l : Filter α
    inst✝¹ : Module 𝕜 E'
    inst✝ : BoundedSMul 𝕜 E'
    c : 𝕜
    hc : Ne c 0
    ⊢ Iff (Asymptotics.IsLittleO l (fun x => HSMul.hSMul c (f' x)) g) (Asymptotics …
  -/
  have cne0 : ‖c‖ ≠ 0 := norm_ne_zero_iff.mpr hc
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    𝕜 : Type u_15
    inst✝⁴ : Norm F
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedDivisionRing 𝕜
    g : α → F
    f' : α → E'
    l : Filter α
    inst✝¹ : Module 𝕜 E'
    inst✝ : BoundedSMul 𝕜 E'
    c : 𝕜
    hc : Ne c 0
    cne0 : Ne (Norm.norm c) 0
    ⊢ Iff (Asymptotics.IsLittleO l (fun x => HSMul.hSMul c (f' x)) g) (Asymptotics …
  -/
  rw [← isLittleO_norm_left]
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    𝕜 : Type u_15
    inst✝⁴ : Norm F
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedDivisionRing 𝕜
    g : α → F
    f' : α → E'
    l : Filter α
    inst✝¹ : Module 𝕜 E'
    inst✝ : BoundedSMul 𝕜 E'
    c : 𝕜
    hc : Ne c 0
    cne0 : Ne (Norm.norm c) 0
    ⊢ Iff (Asymptotics.IsLittleO l (fun x => Norm.norm (HSMul.hSMul c (f' x))) g)  …
  -/
  simp only [norm_smul]
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    𝕜 : Type u_15
    inst✝⁴ : Norm F
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedDivisionRing 𝕜
    g : α → F
    f' : α → E'
    l : Filter α
    inst✝¹ : Module 𝕜 E'
    inst✝ : BoundedSMul 𝕜 E'
    c : 𝕜
    hc : Ne c 0
    cne0 : Ne (Norm.norm c) 0
    ⊢ Iff (Asymptotics.IsLittleO l (fun x => HMul.hMul (Norm.norm c) (Norm.norm (f …
  -/
  rw [isLittleO_const_mul_left_iff cne0, isLittleO_norm_left]
  /-
    🎉 no goals
  -/


theorem isBigO_const_smul_right {c : 𝕜} (hc : c ≠ 0) :
    (f =O[l] fun x => c • f' x) ↔ f =O[l] f' := by
  /-
    α : Type u_1
    E : Type u_3
    E' : Type u_6
    𝕜 : Type u_15
    inst✝⁴ : Norm E
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedDivisionRing 𝕜
    f : α → E
    f' : α → E'
    l : Filter α
    inst✝¹ : Module 𝕜 E'
    inst✝ : BoundedSMul 𝕜 E'
    c : 𝕜
    hc : Ne c 0
    ⊢ Iff (Asymptotics.IsBigO l f fun x => HSMul.hSMul c (f' x)) (Asymptotics.IsBi …
  -/
  have cne0 : ‖c‖ ≠ 0 := norm_ne_zero_iff.mpr hc
  /-
    α : Type u_1
    E : Type u_3
    E' : Type u_6
    𝕜 : Type u_15
    inst✝⁴ : Norm E
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedDivisionRing 𝕜
    f : α → E
    f' : α → E'
    l : Filter α
    inst✝¹ : Module 𝕜 E'
    inst✝ : BoundedSMul 𝕜 E'
    c : 𝕜
    hc : Ne c 0
    cne0 : Ne (Norm.norm c) 0
    ⊢ Iff (Asymptotics.IsBigO l f fun x => HSMul.hSMul c (f' x)) (Asymptotics.IsBi …
  -/
  rw [← isBigO_norm_right]
  /-
    α : Type u_1
    E : Type u_3
    E' : Type u_6
    𝕜 : Type u_15
    inst✝⁴ : Norm E
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedDivisionRing 𝕜
    f : α → E
    f' : α → E'
    l : Filter α
    inst✝¹ : Module 𝕜 E'
    inst✝ : BoundedSMul 𝕜 E'
    c : 𝕜
    hc : Ne c 0
    cne0 : Ne (Norm.norm c) 0
    ⊢ Iff (Asymptotics.IsBigO l f fun x => Norm.norm (HSMul.hSMul c (f' x))) (Asym …
  -/
  simp only [norm_smul]
  /-
    α : Type u_1
    E : Type u_3
    E' : Type u_6
    𝕜 : Type u_15
    inst✝⁴ : Norm E
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedDivisionRing 𝕜
    f : α → E
    f' : α → E'
    l : Filter α
    inst✝¹ : Module 𝕜 E'
    inst✝ : BoundedSMul 𝕜 E'
    c : 𝕜
    hc : Ne c 0
    cne0 : Ne (Norm.norm c) 0
    ⊢ Iff (Asymptotics.IsBigO l f fun x => HMul.hMul (Norm.norm c) (Norm.norm (f'  …
  -/
  rw [isBigO_const_mul_right_iff cne0, isBigO_norm_right]
  /-
    🎉 no goals
  -/


theorem isLittleO_const_smul_right {c : 𝕜} (hc : c ≠ 0) :
    (f =o[l] fun x => c • f' x) ↔ f =o[l] f' := by
  /-
    α : Type u_1
    E : Type u_3
    E' : Type u_6
    𝕜 : Type u_15
    inst✝⁴ : Norm E
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedDivisionRing 𝕜
    f : α → E
    f' : α → E'
    l : Filter α
    inst✝¹ : Module 𝕜 E'
    inst✝ : BoundedSMul 𝕜 E'
    c : 𝕜
    hc : Ne c 0
    ⊢ Iff (Asymptotics.IsLittleO l f fun x => HSMul.hSMul c (f' x)) (Asymptotics.I …
  -/
  have cne0 : ‖c‖ ≠ 0 := norm_ne_zero_iff.mpr hc
  /-
    α : Type u_1
    E : Type u_3
    E' : Type u_6
    𝕜 : Type u_15
    inst✝⁴ : Norm E
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedDivisionRing 𝕜
    f : α → E
    f' : α → E'
    l : Filter α
    inst✝¹ : Module 𝕜 E'
    inst✝ : BoundedSMul 𝕜 E'
    c : 𝕜
    hc : Ne c 0
    cne0 : Ne (Norm.norm c) 0
    ⊢ Iff (Asymptotics.IsLittleO l f fun x => HSMul.hSMul c (f' x)) (Asymptotics.I …
  -/
  rw [← isLittleO_norm_right]
  /-
    α : Type u_1
    E : Type u_3
    E' : Type u_6
    𝕜 : Type u_15
    inst✝⁴ : Norm E
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedDivisionRing 𝕜
    f : α → E
    f' : α → E'
    l : Filter α
    inst✝¹ : Module 𝕜 E'
    inst✝ : BoundedSMul 𝕜 E'
    c : 𝕜
    hc : Ne c 0
    cne0 : Ne (Norm.norm c) 0
    ⊢ Iff (Asymptotics.IsLittleO l f fun x => Norm.norm (HSMul.hSMul c (f' x))) (A …
  -/
  simp only [norm_smul]
  /-
    α : Type u_1
    E : Type u_3
    E' : Type u_6
    𝕜 : Type u_15
    inst✝⁴ : Norm E
    inst✝³ : SeminormedAddCommGroup E'
    inst✝² : NormedDivisionRing 𝕜
    f : α → E
    f' : α → E'
    l : Filter α
    inst✝¹ : Module 𝕜 E'
    inst✝ : BoundedSMul 𝕜 E'
    c : 𝕜
    hc : Ne c 0
    cne0 : Ne (Norm.norm c) 0
    ⊢ Iff (Asymptotics.IsLittleO l f fun x => HMul.hMul (Norm.norm c) (Norm.norm ( …
  -/
  rw [isLittleO_const_mul_right_iff cne0, isLittleO_norm_right]
  /-
    🎉 no goals
  -/


theorem IsBigOWith.smul (h₁ : IsBigOWith c l k₁ k₂) (h₂ : IsBigOWith c' l f' g') :
    IsBigOWith (c * c') l (fun x => k₁ x • f' x) fun x => k₂ x • g' x := by
  /-
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    R : Type u_13
    𝕜' : Type u_16
    inst✝⁷ : SeminormedAddCommGroup E'
    inst✝⁶ : SeminormedAddCommGroup F'
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : NormedDivisionRing 𝕜'
    c c' : Real
    f' : α → E'
    g' : α → F'
    l : Filter α
    inst✝³ : Module R E'
    inst✝² : BoundedSMul R E'
    inst✝¹ : Module 𝕜' F'
    inst✝ : BoundedSMul 𝕜' F'
    k₁ : α → R
    k₂ : α → 𝕜'
    h₁ : Asymptotics.IsBigOWith c l k₁ k₂
    h₂ : Asymptotics.IsBigOWith c' l f' g'
    ⊢ Asymptotics.IsBigOWith (HMul.hMul c c') l (fun x => HSMul.hSMul (k₁ x) (f' x …
  -/
  simp only [IsBigOWith_def] at *
  /-
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    R : Type u_13
    𝕜' : Type u_16
    inst✝⁷ : SeminormedAddCommGroup E'
    inst✝⁶ : SeminormedAddCommGroup F'
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : NormedDivisionRing 𝕜'
    c c' : Real
    f' : α → E'
    g' : α → F'
    l : Filter α
    inst✝³ : Module R E'
    inst✝² : BoundedSMul R E'
    inst✝¹ : Module 𝕜' F'
    inst✝ : BoundedSMul 𝕜' F'
    k₁ : α → R
    k₂ : α → 𝕜'
    h₁ : Filter.Eventually (fun x => LE.le (Norm.norm (k₁ x)) (HMul.hMul c (Norm.n …
    h₂ : Filter.Eventually (fun x => LE.le (Norm.norm (f' x)) (HMul.hMul c' (Norm. …
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HSMul.hSMul (k₁ x) (f' x))) (H …
  -/
  filter_upwards [h₁, h₂] with _ hx₁ hx₂
  /-
    case h
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    R : Type u_13
    𝕜' : Type u_16
    inst✝⁷ : SeminormedAddCommGroup E'
    inst✝⁶ : SeminormedAddCommGroup F'
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : NormedDivisionRing 𝕜'
    c c' : Real
    f' : α → E'
    g' : α → F'
    l : Filter α
    inst✝³ : Module R E'
    inst✝² : BoundedSMul R E'
    inst✝¹ : Module 𝕜' F'
    inst✝ : BoundedSMul 𝕜' F'
    k₁ : α → R
    k₂ : α → 𝕜'
    h₁ : Filter.Eventually (fun x => LE.le (Norm.norm (k₁ x)) (HMul.hMul c (Norm.n …
    h₂ : Filter.Eventually (fun x => LE.le (Norm.norm (f' x)) (HMul.hMul c' (Norm. …
    a✝ : α
    hx₁ : LE.le (Norm.norm (k₁ a✝)) (HMul.hMul c (Norm.norm (k₂ a✝)))
    hx₂ : LE.le (Norm.norm (f' a✝)) (HMul.hMul c' (Norm.norm (g' a✝)))
    ⊢ LE.le (Norm.norm (HSMul.hSMul (k₁ a✝) (f' a✝))) (HMul.hMul (HMul.hMul c c')  …
  -/
  apply le_trans (norm_smul_le _ _)
  /-
    case h
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    R : Type u_13
    𝕜' : Type u_16
    inst✝⁷ : SeminormedAddCommGroup E'
    inst✝⁶ : SeminormedAddCommGroup F'
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : NormedDivisionRing 𝕜'
    c c' : Real
    f' : α → E'
    g' : α → F'
    l : Filter α
    inst✝³ : Module R E'
    inst✝² : BoundedSMul R E'
    inst✝¹ : Module 𝕜' F'
    inst✝ : BoundedSMul 𝕜' F'
    k₁ : α → R
    k₂ : α → 𝕜'
    h₁ : Filter.Eventually (fun x => LE.le (Norm.norm (k₁ x)) (HMul.hMul c (Norm.n …
    h₂ : Filter.Eventually (fun x => LE.le (Norm.norm (f' x)) (HMul.hMul c' (Norm. …
    a✝ : α
    hx₁ : LE.le (Norm.norm (k₁ a✝)) (HMul.hMul c (Norm.norm (k₂ a✝)))
    hx₂ : LE.le (Norm.norm (f' a✝)) (HMul.hMul c' (Norm.norm (g' a✝)))
    ⊢ LE.le (HMul.hMul (Norm.norm (k₁ a✝)) (Norm.norm (f' a✝))) (HMul.hMul (HMul.h …
  -/
  convert mul_le_mul hx₁ hx₂ (norm_nonneg _) (le_trans (norm_nonneg _) hx₁) using 1
  /-
    case h.e'_4
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    R : Type u_13
    𝕜' : Type u_16
    inst✝⁷ : SeminormedAddCommGroup E'
    inst✝⁶ : SeminormedAddCommGroup F'
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : NormedDivisionRing 𝕜'
    c c' : Real
    f' : α → E'
    g' : α → F'
    l : Filter α
    inst✝³ : Module R E'
    inst✝² : BoundedSMul R E'
    inst✝¹ : Module 𝕜' F'
    inst✝ : BoundedSMul 𝕜' F'
    k₁ : α → R
    k₂ : α → 𝕜'
    h₁ : Filter.Eventually (fun x => LE.le (Norm.norm (k₁ x)) (HMul.hMul c (Norm.n …
    h₂ : Filter.Eventually (fun x => LE.le (Norm.norm (f' x)) (HMul.hMul c' (Norm. …
    a✝ : α
    hx₁ : LE.le (Norm.norm (k₁ a✝)) (HMul.hMul c (Norm.norm (k₂ a✝)))
    hx₂ : LE.le (Norm.norm (f' a✝)) (HMul.hMul c' (Norm.norm (g' a✝)))
    ⊢ Eq (HMul.hMul (HMul.hMul c c') (Norm.norm (HSMul.hSMul (k₂ a✝) (g' a✝)))) (H …
  -/
  rw [norm_smul, mul_mul_mul_comm]
  /-
    🎉 no goals
  -/


theorem IsBigO.smul (h₁ : k₁ =O[l] k₂) (h₂ : f' =O[l] g') :
    (fun x => k₁ x • f' x) =O[l] fun x => k₂ x • g' x := by
  /-
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    R : Type u_13
    𝕜' : Type u_16
    inst✝⁷ : SeminormedAddCommGroup E'
    inst✝⁶ : SeminormedAddCommGroup F'
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : NormedDivisionRing 𝕜'
    f' : α → E'
    g' : α → F'
    l : Filter α
    inst✝³ : Module R E'
    inst✝² : BoundedSMul R E'
    inst✝¹ : Module 𝕜' F'
    inst✝ : BoundedSMul 𝕜' F'
    k₁ : α → R
    k₂ : α → 𝕜'
    h₁ : Asymptotics.IsBigO l k₁ k₂
    h₂ : Asymptotics.IsBigO l f' g'
    ⊢ Asymptotics.IsBigO l (fun x => HSMul.hSMul (k₁ x) (f' x)) fun x => HSMul.hSM …
  -/
  obtain ⟨c₁, h₁⟩ := h₁.isBigOWith
  /-
    case intro
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    R : Type u_13
    𝕜' : Type u_16
    inst✝⁷ : SeminormedAddCommGroup E'
    inst✝⁶ : SeminormedAddCommGroup F'
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : NormedDivisionRing 𝕜'
    f' : α → E'
    g' : α → F'
    l : Filter α
    inst✝³ : Module R E'
    inst✝² : BoundedSMul R E'
    inst✝¹ : Module 𝕜' F'
    inst✝ : BoundedSMul 𝕜' F'
    k₁ : α → R
    k₂ : α → 𝕜'
    h₁✝ : Asymptotics.IsBigO l k₁ k₂
    h₂ : Asymptotics.IsBigO l f' g'
    c₁ : Real
    h₁ : Asymptotics.IsBigOWith c₁ l k₁ k₂
    ⊢ Asymptotics.IsBigO l (fun x => HSMul.hSMul (k₁ x) (f' x)) fun x => HSMul.hSM …
  -/
  obtain ⟨c₂, h₂⟩ := h₂.isBigOWith
  /-
    case intro.intro
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    R : Type u_13
    𝕜' : Type u_16
    inst✝⁷ : SeminormedAddCommGroup E'
    inst✝⁶ : SeminormedAddCommGroup F'
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : NormedDivisionRing 𝕜'
    f' : α → E'
    g' : α → F'
    l : Filter α
    inst✝³ : Module R E'
    inst✝² : BoundedSMul R E'
    inst✝¹ : Module 𝕜' F'
    inst✝ : BoundedSMul 𝕜' F'
    k₁ : α → R
    k₂ : α → 𝕜'
    h₁✝ : Asymptotics.IsBigO l k₁ k₂
    h₂✝ : Asymptotics.IsBigO l f' g'
    c₁ : Real
    h₁ : Asymptotics.IsBigOWith c₁ l k₁ k₂
    c₂ : Real
    h₂ : Asymptotics.IsBigOWith c₂ l f' g'
    ⊢ Asymptotics.IsBigO l (fun x => HSMul.hSMul (k₁ x) (f' x)) fun x => HSMul.hSM …
  -/
  exact (h₁.smul h₂).isBigO
  /-
    🎉 no goals
  -/


theorem IsBigO.smul_isLittleO (h₁ : k₁ =O[l] k₂) (h₂ : f' =o[l] g') :
    (fun x => k₁ x • f' x) =o[l] fun x => k₂ x • g' x := by
  /-
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    R : Type u_13
    𝕜' : Type u_16
    inst✝⁷ : SeminormedAddCommGroup E'
    inst✝⁶ : SeminormedAddCommGroup F'
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : NormedDivisionRing 𝕜'
    f' : α → E'
    g' : α → F'
    l : Filter α
    inst✝³ : Module R E'
    inst✝² : BoundedSMul R E'
    inst✝¹ : Module 𝕜' F'
    inst✝ : BoundedSMul 𝕜' F'
    k₁ : α → R
    k₂ : α → 𝕜'
    h₁ : Asymptotics.IsBigO l k₁ k₂
    h₂ : Asymptotics.IsLittleO l f' g'
    ⊢ Asymptotics.IsLittleO l (fun x => HSMul.hSMul (k₁ x) (f' x)) fun x => HSMul. …
  -/
  simp only [IsLittleO_def] at *
  /-
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    R : Type u_13
    𝕜' : Type u_16
    inst✝⁷ : SeminormedAddCommGroup E'
    inst✝⁶ : SeminormedAddCommGroup F'
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : NormedDivisionRing 𝕜'
    f' : α → E'
    g' : α → F'
    l : Filter α
    inst✝³ : Module R E'
    inst✝² : BoundedSMul R E'
    inst✝¹ : Module 𝕜' F'
    inst✝ : BoundedSMul 𝕜' F'
    k₁ : α → R
    k₂ : α → 𝕜'
    h₁ : Asymptotics.IsBigO l k₁ k₂
    h₂ : ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l f' g'
    ⊢ ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l (fun x => HSMul.hSMul ( …
  -/
  intro c cpos
  /-
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    R : Type u_13
    𝕜' : Type u_16
    inst✝⁷ : SeminormedAddCommGroup E'
    inst✝⁶ : SeminormedAddCommGroup F'
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : NormedDivisionRing 𝕜'
    f' : α → E'
    g' : α → F'
    l : Filter α
    inst✝³ : Module R E'
    inst✝² : BoundedSMul R E'
    inst✝¹ : Module 𝕜' F'
    inst✝ : BoundedSMul 𝕜' F'
    k₁ : α → R
    k₂ : α → 𝕜'
    h₁ : Asymptotics.IsBigO l k₁ k₂
    h₂ : ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l f' g'
    c : Real
    cpos : LT.lt 0 c
    ⊢ Asymptotics.IsBigOWith c l (fun x => HSMul.hSMul (k₁ x) (f' x)) fun x => HSM …
  -/
  rcases h₁.exists_pos with ⟨c', c'pos, hc'⟩
  /-
    case intro.intro
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    R : Type u_13
    𝕜' : Type u_16
    inst✝⁷ : SeminormedAddCommGroup E'
    inst✝⁶ : SeminormedAddCommGroup F'
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : NormedDivisionRing 𝕜'
    f' : α → E'
    g' : α → F'
    l : Filter α
    inst✝³ : Module R E'
    inst✝² : BoundedSMul R E'
    inst✝¹ : Module 𝕜' F'
    inst✝ : BoundedSMul 𝕜' F'
    k₁ : α → R
    k₂ : α → 𝕜'
    h₁ : Asymptotics.IsBigO l k₁ k₂
    h₂ : ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l f' g'
    c : Real
    cpos : LT.lt 0 c
    c' : Real
    c'pos : GT.gt c' 0
    hc' : Asymptotics.IsBigOWith c' l k₁ k₂
    ⊢ Asymptotics.IsBigOWith c l (fun x => HSMul.hSMul (k₁ x) (f' x)) fun x => HSM …
  -/
  exact (hc'.smul (h₂ (div_pos cpos c'pos))).congr_const (mul_div_cancel₀ _ (ne_of_gt c'pos))
  /-
    🎉 no goals
  -/


theorem IsLittleO.smul_isBigO (h₁ : k₁ =o[l] k₂) (h₂ : f' =O[l] g') :
    (fun x => k₁ x • f' x) =o[l] fun x => k₂ x • g' x := by
  /-
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    R : Type u_13
    𝕜' : Type u_16
    inst✝⁷ : SeminormedAddCommGroup E'
    inst✝⁶ : SeminormedAddCommGroup F'
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : NormedDivisionRing 𝕜'
    f' : α → E'
    g' : α → F'
    l : Filter α
    inst✝³ : Module R E'
    inst✝² : BoundedSMul R E'
    inst✝¹ : Module 𝕜' F'
    inst✝ : BoundedSMul 𝕜' F'
    k₁ : α → R
    k₂ : α → 𝕜'
    h₁ : Asymptotics.IsLittleO l k₁ k₂
    h₂ : Asymptotics.IsBigO l f' g'
    ⊢ Asymptotics.IsLittleO l (fun x => HSMul.hSMul (k₁ x) (f' x)) fun x => HSMul. …
  -/
  simp only [IsLittleO_def] at *
  /-
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    R : Type u_13
    𝕜' : Type u_16
    inst✝⁷ : SeminormedAddCommGroup E'
    inst✝⁶ : SeminormedAddCommGroup F'
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : NormedDivisionRing 𝕜'
    f' : α → E'
    g' : α → F'
    l : Filter α
    inst✝³ : Module R E'
    inst✝² : BoundedSMul R E'
    inst✝¹ : Module 𝕜' F'
    inst✝ : BoundedSMul 𝕜' F'
    k₁ : α → R
    k₂ : α → 𝕜'
    h₂ : Asymptotics.IsBigO l f' g'
    h₁ : ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l k₁ k₂
    ⊢ ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l (fun x => HSMul.hSMul ( …
  -/
  intro c cpos
  /-
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    R : Type u_13
    𝕜' : Type u_16
    inst✝⁷ : SeminormedAddCommGroup E'
    inst✝⁶ : SeminormedAddCommGroup F'
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : NormedDivisionRing 𝕜'
    f' : α → E'
    g' : α → F'
    l : Filter α
    inst✝³ : Module R E'
    inst✝² : BoundedSMul R E'
    inst✝¹ : Module 𝕜' F'
    inst✝ : BoundedSMul 𝕜' F'
    k₁ : α → R
    k₂ : α → 𝕜'
    h₂ : Asymptotics.IsBigO l f' g'
    h₁ : ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l k₁ k₂
    c : Real
    cpos : LT.lt 0 c
    ⊢ Asymptotics.IsBigOWith c l (fun x => HSMul.hSMul (k₁ x) (f' x)) fun x => HSM …
  -/
  rcases h₂.exists_pos with ⟨c', c'pos, hc'⟩
  /-
    case intro.intro
    α : Type u_1
    E' : Type u_6
    F' : Type u_7
    R : Type u_13
    𝕜' : Type u_16
    inst✝⁷ : SeminormedAddCommGroup E'
    inst✝⁶ : SeminormedAddCommGroup F'
    inst✝⁵ : SeminormedRing R
    inst✝⁴ : NormedDivisionRing 𝕜'
    f' : α → E'
    g' : α → F'
    l : Filter α
    inst✝³ : Module R E'
    inst✝² : BoundedSMul R E'
    inst✝¹ : Module 𝕜' F'
    inst✝ : BoundedSMul 𝕜' F'
    k₁ : α → R
    k₂ : α → 𝕜'
    h₂ : Asymptotics.IsBigO l f' g'
    h₁ : ∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c l k₁ k₂
    c : Real
    cpos : LT.lt 0 c
    c' : Real
    c'pos : GT.gt c' 0
    hc' : Asymptotics.IsBigOWith c' l f' g'
    ⊢ Asymptotics.IsBigOWith c l (fun x => HSMul.hSMul (k₁ x) (f' x)) fun x => HSM …
  -/
  exact ((h₁ (div_pos cpos c'pos)).smul hc').congr_const (div_mul_cancel₀ _ (ne_of_gt c'pos))
  /-
    🎉 no goals
  -/


theorem IsLittleO.smul (h₁ : k₁ =o[l] k₂) (h₂ : f' =o[l] g') :
    (fun x => k₁ x • f' x) =o[l] fun x => k₂ x • g' x :=
  h₁.smul_isBigO h₂.isBigO


theorem IsBigOWith.sum (h : ∀ i ∈ s, IsBigOWith (C i) l (A i) g) :
    IsBigOWith (∑ i ∈ s, C i) l (fun x => ∑ i ∈ s, A i x) g := by
  classical
  induction' s using Finset.induction_on with i s is IH
  · simp only [isBigOWith_zero', Finset.sum_empty, forall_true_iff]
  · simp only [is, Finset.sum_insert, not_false_iff]
    exact (h _ (Finset.mem_insert_self i s)).add (IH fun j hj => h _ (Finset.mem_insert_of_mem hj))


theorem IsBigO.sum (h : ∀ i ∈ s, A i =O[l] g) : (fun x => ∑ i ∈ s, A i x) =O[l] g := by
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    inst✝¹ : Norm F
    inst✝ : SeminormedAddCommGroup E'
    g : α → F
    l : Filter α
    ι : Type u_17
    A : ι → α → E'
    s : Finset ι
    h : ∀ (i : ι), Membership.mem s i → Asymptotics.IsBigO l (A i) g
    ⊢ Asymptotics.IsBigO l (fun x => s.sum fun i => A i x) g
  -/
  simp only [IsBigO_def] at *
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    inst✝¹ : Norm F
    inst✝ : SeminormedAddCommGroup E'
    g : α → F
    l : Filter α
    ι : Type u_17
    A : ι → α → E'
    s : Finset ι
    h : ∀ (i : ι), Membership.mem s i → Exists fun c => Asymptotics.IsBigOWith c l …
    ⊢ Exists fun c => Asymptotics.IsBigOWith c l (fun x => s.sum fun i => A i x) g
  -/
  choose! C hC using h
  /-
    α : Type u_1
    F : Type u_4
    E' : Type u_6
    inst✝¹ : Norm F
    inst✝ : SeminormedAddCommGroup E'
    g : α → F
    l : Filter α
    ι : Type u_17
    A : ι → α → E'
    s : Finset ι
    C : ι → Real
    hC : ∀ (i : ι), Membership.mem s i → Asymptotics.IsBigOWith (C i) l (A i) g
    ⊢ Exists fun c => Asymptotics.IsBigOWith c l (fun x => s.sum fun i => A i x) g
  -/
  exact ⟨_, IsBigOWith.sum hC⟩
  /-
    🎉 no goals
  -/


theorem IsLittleO.sum (h : ∀ i ∈ s, A i =o[l] g') : (fun x => ∑ i ∈ s, A i x) =o[l] g' := by
  classical
  induction' s using Finset.induction_on with i s is IH
  · simp only [isLittleO_zero, Finset.sum_empty, forall_true_iff]
  · simp only [is, Finset.sum_insert, not_false_iff]
    exact (h _ (Finset.mem_insert_self i s)).add (IH fun j hj => h _ (Finset.mem_insert_of_mem hj))


theorem IsLittleO.tendsto_div_nhds_zero {f g : α → 𝕜} (h : f =o[l] g) :
    Tendsto (fun x => f x / g x) l (𝓝 0) :=
  (isLittleO_one_iff 𝕜).mp <| by
    calc
      (fun x => f x / g x) =o[l] fun x => g x / g x := by
        simpa only [div_eq_mul_inv] using h.mul_isBigO (isBigO_refl _ _)
      _ =O[l] fun _x => (1 : 𝕜) := isBigO_of_le _ fun x => by simp [div_self_le_one]


theorem IsLittleO.tendsto_inv_smul_nhds_zero [Module 𝕜 E'] [BoundedSMul 𝕜 E']
    {f : α → E'} {g : α → 𝕜}
    {l : Filter α} (h : f =o[l] g) : Tendsto (fun x => (g x)⁻¹ • f x) l (𝓝 0) := by
  simpa only [div_eq_inv_mul, ← norm_inv, ← norm_smul, ← tendsto_zero_iff_norm_tendsto_zero] using
    h.norm_norm.tendsto_div_nhds_zero


theorem isLittleO_iff_tendsto' {f g : α → 𝕜} (hgf : ∀ᶠ x in l, g x = 0 → f x = 0) :
    f =o[l] g ↔ Tendsto (fun x => f x / g x) l (𝓝 0) :=
  ⟨IsLittleO.tendsto_div_nhds_zero, fun h =>
    (((isLittleO_one_iff _).mpr h).mul_isBigO (isBigO_refl g l)).congr'
      (hgf.mono fun _x => div_mul_cancel_of_imp) (Eventually.of_forall fun _x => one_mul _)⟩


theorem isLittleO_iff_tendsto {f g : α → 𝕜} (hgf : ∀ x, g x = 0 → f x = 0) :
    f =o[l] g ↔ Tendsto (fun x => f x / g x) l (𝓝 0) :=
  isLittleO_iff_tendsto' (Eventually.of_forall hgf)


alias ⟨_, isLittleO_of_tendsto'⟩ := isLittleO_iff_tendsto'


alias ⟨_, isLittleO_of_tendsto⟩ := isLittleO_iff_tendsto


theorem isLittleO_const_left_of_ne {c : E''} (hc : c ≠ 0) :
    (fun _x => c) =o[l] g ↔ Tendsto (fun x => ‖g x‖) l atTop := by
  /-
    α : Type u_1
    F : Type u_4
    E'' : Type u_9
    inst✝¹ : Norm F
    inst✝ : NormedAddCommGroup E''
    g : α → F
    l : Filter α
    c : E''
    hc : Ne c 0
    ⊢ Iff (Asymptotics.IsLittleO l (fun _x => c) g) (Filter.Tendsto (fun x => Norm …
  -/
  simp only [← isLittleO_one_left_iff ℝ]
  exact ⟨(isBigO_const_const (1 : ℝ) hc l).trans_isLittleO,
    (isBigO_const_one ℝ c l).trans_isLittleO⟩


@[simp]
theorem isLittleO_const_left {c : E''} :
    (fun _x => c) =o[l] g'' ↔ c = 0 ∨ Tendsto (norm ∘ g'') l atTop := by
  /-
    α : Type u_1
    E'' : Type u_9
    F'' : Type u_10
    inst✝¹ : NormedAddCommGroup E''
    inst✝ : NormedAddCommGroup F''
    g'' : α → F''
    l : Filter α
    c : E''
    ⊢ Iff (Asymptotics.IsLittleO l (fun _x => c) g'') (Or (Eq c 0) (Filter.Tendsto …
  -/
  rcases eq_or_ne c 0 with (rfl | hc)
    /-
      case inl
      α : Type u_1
      E'' : Type u_9
      F'' : Type u_10
      inst✝¹ : NormedAddCommGroup E''
      inst✝ : NormedAddCommGroup F''
      g'' : α → F''
      l : Filter α
      ⊢ Iff (Asymptotics.IsLittleO l (fun _x => 0) g'') (Or (Eq 0 0) (Filter.Tendsto …
    -/
  · simp only [isLittleO_zero, eq_self_iff_true, true_or]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      E'' : Type u_9
      F'' : Type u_10
      inst✝¹ : NormedAddCommGroup E''
      inst✝ : NormedAddCommGroup F''
      g'' : α → F''
      l : Filter α
      c : E''
      hc : Ne c 0
      ⊢ Iff (Asymptotics.IsLittleO l (fun _x => c) g'') (Or (Eq c 0) (Filter.Tendsto …
    -/
  · simp only [hc, false_or, isLittleO_const_left_of_ne hc]; rfl
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp 1001] -- Porting note: increase priority so that this triggers before `isLittleO_const_left`
theorem isLittleO_const_const_iff [NeBot l] {d : E''} {c : F''} :
    ((fun _x => d) =o[l] fun _x => c) ↔ d = 0 := by
  have : ¬Tendsto (Function.const α ‖c‖) l atTop :=
    not_tendsto_atTop_of_tendsto_nhds tendsto_const_nhds
  /-
    α : Type u_1
    E'' : Type u_9
    F'' : Type u_10
    inst✝² : NormedAddCommGroup E''
    inst✝¹ : NormedAddCommGroup F''
    l : Filter α
    inst✝ : l.NeBot
    d : E''
    c : F''
    this : Not (Filter.Tendsto (Function.const α (Norm.norm c)) l Filter.atTop)
    ⊢ Iff (Asymptotics.IsLittleO l (fun _x => d) fun _x => c) (Eq d 0)
  -/
  simp only [isLittleO_const_left, or_iff_left_iff_imp]
  /-
    α : Type u_1
    E'' : Type u_9
    F'' : Type u_10
    inst✝² : NormedAddCommGroup E''
    inst✝¹ : NormedAddCommGroup F''
    l : Filter α
    inst✝ : l.NeBot
    d : E''
    c : F''
    this : Not (Filter.Tendsto (Function.const α (Norm.norm c)) l Filter.atTop)
    ⊢ Filter.Tendsto (Function.comp Norm.norm fun _x => c) l Filter.atTop → Eq d 0
  -/
  exact fun h => (this h).elim
  /-
    🎉 no goals
  -/


@[simp]
theorem isLittleO_pure {x} : f'' =o[pure x] g'' ↔ f'' x = 0 :=
  calc
    f'' =o[pure x] g'' ↔ (fun _y : α => f'' x) =o[pure x] fun _ => g'' x := isLittleO_congr rfl rfl
    _ ↔ f'' x = 0 := isLittleO_const_const_iff


theorem isLittleO_const_id_cobounded (c : F'') :
    (fun _ => c) =o[Bornology.cobounded E''] id :=
  isLittleO_const_left.2 <| .inr tendsto_norm_cobounded_atTop


theorem isLittleO_const_id_atTop (c : E'') : (fun _x : ℝ => c) =o[atTop] id :=
  isLittleO_const_left.2 <| Or.inr tendsto_abs_atTop_atTop


theorem isLittleO_const_id_atBot (c : E'') : (fun _x : ℝ => c) =o[atBot] id :=
  isLittleO_const_left.2 <| Or.inr tendsto_abs_atBot_atTop


theorem IsBigOWith.eventually_mul_div_cancel (h : IsBigOWith c l u v) : u / v * v =ᶠ[l] u :=
                                                                         /-
                                                                           α : Type u_1
                                                                           𝕜 : Type u_15
                                                                           inst✝ : NormedDivisionRing 𝕜
                                                                           c : Real
                                                                           l : Filter α
                                                                           u v : α → 𝕜
                                                                           h : Asymptotics.IsBigOWith c l u v
                                                                           y : α
                                                                           hy : LE.le (Norm.norm (u y)) (HMul.hMul c (Norm.norm (v y)))
                                                                           hv : Eq (v y) 0
                                                                           ⊢ Eq (u y) 0
                                                                         -/
  Eventually.mono h.bound fun y hy => div_mul_cancel_of_imp fun hv => by simpa [hv] using hy
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- If `u = O(v)` along `l`, then `(u / v) * v = u` eventually at `l`. -/
theorem IsBigO.eventually_mul_div_cancel (h : u =O[l] v) : u / v * v =ᶠ[l] u :=
  let ⟨_c, hc⟩ := h.isBigOWith
  hc.eventually_mul_div_cancel


/-- If `u = o(v)` along `l`, then `(u / v) * v = u` eventually at `l`. -/
theorem IsLittleO.eventually_mul_div_cancel (h : u =o[l] v) : u / v * v =ᶠ[l] u :=
  (h.forall_isBigOWith zero_lt_one).eventually_mul_div_cancel


/-- If `‖φ‖` is eventually bounded by `c`, and `u =ᶠ[l] φ * v`, then we have `IsBigOWith c u v l`.
    This does not require any assumptions on `c`, which is why we keep this version along with
    `IsBigOWith_iff_exists_eq_mul`. -/
theorem isBigOWith_of_eq_mul {u v : α → R} (φ : α → R) (hφ : ∀ᶠ x in l, ‖φ x‖ ≤ c)
    (h : u =ᶠ[l] φ * v) :
    IsBigOWith c l u v := by
  /-
    α : Type u_1
    R : Type u_13
    inst✝ : SeminormedRing R
    c : Real
    l : Filter α
    u v φ : α → R
    hφ : Filter.Eventually (fun x => LE.le (Norm.norm (φ x)) c) l
    h : l.EventuallyEq u (HMul.hMul φ v)
    ⊢ Asymptotics.IsBigOWith c l u v
  -/
  simp only [IsBigOWith_def]
  /-
    α : Type u_1
    R : Type u_13
    inst✝ : SeminormedRing R
    c : Real
    l : Filter α
    u v φ : α → R
    hφ : Filter.Eventually (fun x => LE.le (Norm.norm (φ x)) c) l
    h : l.EventuallyEq u (HMul.hMul φ v)
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (u x)) (HMul.hMul c (Norm.norm  …
  -/
  refine h.symm.rw (fun x a => ‖a‖ ≤ c * ‖v x‖) (hφ.mono fun x hx => ?_)
  /-
    α : Type u_1
    R : Type u_13
    inst✝ : SeminormedRing R
    c : Real
    l : Filter α
    u v φ : α → R
    hφ : Filter.Eventually (fun x => LE.le (Norm.norm (φ x)) c) l
    h : l.EventuallyEq u (HMul.hMul φ v)
    x : α
    hx : LE.le (Norm.norm (φ x)) c
    ⊢ (fun x a => LE.le (Norm.norm a) (HMul.hMul c (Norm.norm (v x)))) x (HMul.hMu …
  -/
  simp only [Pi.mul_apply]
  /-
    α : Type u_1
    R : Type u_13
    inst✝ : SeminormedRing R
    c : Real
    l : Filter α
    u v φ : α → R
    hφ : Filter.Eventually (fun x => LE.le (Norm.norm (φ x)) c) l
    h : l.EventuallyEq u (HMul.hMul φ v)
    x : α
    hx : LE.le (Norm.norm (φ x)) c
    ⊢ LE.le (Norm.norm (HMul.hMul (φ x) (v x))) (HMul.hMul c (Norm.norm (v x)))
  -/
  refine (norm_mul_le _ _).trans ?_
  /-
    α : Type u_1
    R : Type u_13
    inst✝ : SeminormedRing R
    c : Real
    l : Filter α
    u v φ : α → R
    hφ : Filter.Eventually (fun x => LE.le (Norm.norm (φ x)) c) l
    h : l.EventuallyEq u (HMul.hMul φ v)
    x : α
    hx : LE.le (Norm.norm (φ x)) c
    ⊢ LE.le (HMul.hMul (Norm.norm (φ x)) (Norm.norm (v x))) (HMul.hMul c (Norm.nor …
  -/
  gcongr
  /-
    🎉 no goals
  -/


theorem isBigOWith_iff_exists_eq_mul (hc : 0 ≤ c) :
    IsBigOWith c l u v ↔ ∃ φ : α → 𝕜, (∀ᶠ x in l, ‖φ x‖ ≤ c) ∧ u =ᶠ[l] φ * v := by
  /-
    α : Type u_1
    𝕜 : Type u_15
    inst✝ : NormedDivisionRing 𝕜
    c : Real
    l : Filter α
    u v : α → 𝕜
    hc : LE.le 0 c
    ⊢ Iff (Asymptotics.IsBigOWith c l u v) (Exists fun φ => And (Filter.Eventually …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      c : Real
      l : Filter α
      u v : α → 𝕜
      hc : LE.le 0 c
      ⊢ Asymptotics.IsBigOWith c l u v → Exists fun φ => And (Filter.Eventually (fun …
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      c : Real
      l : Filter α
      u v : α → 𝕜
      hc : LE.le 0 c
      h : Asymptotics.IsBigOWith c l u v
      ⊢ Exists fun φ => And (Filter.Eventually (fun x => LE.le (Norm.norm (φ x)) c)  …
    -/
    use fun x => u x / v x
    /-
      case h
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      c : Real
      l : Filter α
      u v : α → 𝕜
      hc : LE.le 0 c
      h : Asymptotics.IsBigOWith c l u v
      ⊢ And (Filter.Eventually (fun x => LE.le (Norm.norm ((fun x => HDiv.hDiv (u x) …
    -/
    refine ⟨Eventually.mono h.bound fun y hy => ?_, h.eventually_mul_div_cancel.symm⟩
    /-
      case h
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      c : Real
      l : Filter α
      u v : α → 𝕜
      hc : LE.le 0 c
      h : Asymptotics.IsBigOWith c l u v
      y : α
      hy : LE.le (Norm.norm (u y)) (HMul.hMul c (Norm.norm (v y)))
      ⊢ LE.le (Norm.norm ((fun x => HDiv.hDiv (u x) (v x)) y)) c
    -/
    simpa using div_le_of_le_mul₀ (norm_nonneg _) hc hy
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      c : Real
      l : Filter α
      u v : α → 𝕜
      hc : LE.le 0 c
      ⊢ (Exists fun φ => And (Filter.Eventually (fun x => LE.le (Norm.norm (φ x)) c) …
    -/
  · rintro ⟨φ, hφ, h⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      c : Real
      l : Filter α
      u v : α → 𝕜
      hc : LE.le 0 c
      φ : α → 𝕜
      hφ : Filter.Eventually (fun x => LE.le (Norm.norm (φ x)) c) l
      h : l.EventuallyEq u (HMul.hMul φ v)
      ⊢ Asymptotics.IsBigOWith c l u v
    -/
    exact isBigOWith_of_eq_mul φ hφ h
    /-
      🎉 no goals
    -/


theorem IsBigOWith.exists_eq_mul (h : IsBigOWith c l u v) (hc : 0 ≤ c) :
    ∃ φ : α → 𝕜, (∀ᶠ x in l, ‖φ x‖ ≤ c) ∧ u =ᶠ[l] φ * v :=
  (isBigOWith_iff_exists_eq_mul hc).mp h


theorem isBigO_iff_exists_eq_mul :
    u =O[l] v ↔ ∃ φ : α → 𝕜, l.IsBoundedUnder (· ≤ ·) (norm ∘ φ) ∧ u =ᶠ[l] φ * v := by
  /-
    α : Type u_1
    𝕜 : Type u_15
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    u v : α → 𝕜
    ⊢ Iff (Asymptotics.IsBigO l u v) (Exists fun φ => And (Filter.IsBoundedUnder ( …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      l : Filter α
      u v : α → 𝕜
      ⊢ Asymptotics.IsBigO l u v → Exists fun φ => And (Filter.IsBoundedUnder (fun x …
    -/
  · rintro h
    /-
      case mp
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      l : Filter α
      u v : α → 𝕜
      h : Asymptotics.IsBigO l u v
      ⊢ Exists fun φ => And (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l (Fun …
    -/
    rcases h.exists_nonneg with ⟨c, hnnc, hc⟩
    /-
      case mp.intro.intro
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      l : Filter α
      u v : α → 𝕜
      h : Asymptotics.IsBigO l u v
      c : Real
      hnnc : GE.ge c 0
      hc : Asymptotics.IsBigOWith c l u v
      ⊢ Exists fun φ => And (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l (Fun …
    -/
    rcases hc.exists_eq_mul hnnc with ⟨φ, hφ, huvφ⟩
    /-
      case mp.intro.intro.intro.intro
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      l : Filter α
      u v : α → 𝕜
      h : Asymptotics.IsBigO l u v
      c : Real
      hnnc : GE.ge c 0
      hc : Asymptotics.IsBigOWith c l u v
      φ : α → 𝕜
      hφ : Filter.Eventually (fun x => LE.le (Norm.norm (φ x)) c) l
      huvφ : l.EventuallyEq u (HMul.hMul φ v)
      ⊢ Exists fun φ => And (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l (Fun …
    -/
    exact ⟨φ, ⟨c, hφ⟩, huvφ⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      l : Filter α
      u v : α → 𝕜
      ⊢ (Exists fun φ => And (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l (Fu …
    -/
  · rintro ⟨φ, ⟨c, hφ⟩, huvφ⟩
    /-
      case mpr.intro.intro.intro
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      l : Filter α
      u v φ : α → 𝕜
      huvφ : l.EventuallyEq u (HMul.hMul φ v)
      c : Real
      hφ : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x c) (Filter.map ( …
      ⊢ Asymptotics.IsBigO l u v
    -/
    exact isBigO_iff_isBigOWith.2 ⟨c, isBigOWith_of_eq_mul φ hφ huvφ⟩
    /-
      🎉 no goals
    -/


alias ⟨IsBigO.exists_eq_mul, _⟩ := isBigO_iff_exists_eq_mul


theorem isLittleO_iff_exists_eq_mul :
    u =o[l] v ↔ ∃ φ : α → 𝕜, Tendsto φ l (𝓝 0) ∧ u =ᶠ[l] φ * v := by
  /-
    α : Type u_1
    𝕜 : Type u_15
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    u v : α → 𝕜
    ⊢ Iff (Asymptotics.IsLittleO l u v) (Exists fun φ => And (Filter.Tendsto φ l ( …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      l : Filter α
      u v : α → 𝕜
      ⊢ Asymptotics.IsLittleO l u v → Exists fun φ => And (Filter.Tendsto φ l (nhds  …
    -/
  · exact fun h => ⟨fun x => u x / v x, h.tendsto_div_nhds_zero, h.eventually_mul_div_cancel.symm⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      l : Filter α
      u v : α → 𝕜
      ⊢ (Exists fun φ => And (Filter.Tendsto φ l (nhds 0)) (l.EventuallyEq u (HMul.h …
    -/
  · simp only [IsLittleO_def]
    /-
      case mpr
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      l : Filter α
      u v : α → 𝕜
      ⊢ (Exists fun φ => And (Filter.Tendsto φ l (nhds 0)) (l.EventuallyEq u (HMul.h …
    -/
    rintro ⟨φ, hφ, huvφ⟩ c hpos
    /-
      case mpr.intro.intro
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      l : Filter α
      u v φ : α → 𝕜
      hφ : Filter.Tendsto φ l (nhds 0)
      huvφ : l.EventuallyEq u (HMul.hMul φ v)
      c : Real
      hpos : LT.lt 0 c
      ⊢ Asymptotics.IsBigOWith c l u v
    -/
    rw [NormedAddCommGroup.tendsto_nhds_zero] at hφ
    /-
      case mpr.intro.intro
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      l : Filter α
      u v φ : α → 𝕜
      hφ : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Norm.norm (φ …
      huvφ : l.EventuallyEq u (HMul.hMul φ v)
      c : Real
      hpos : LT.lt 0 c
      ⊢ Asymptotics.IsBigOWith c l u v
    -/
    exact isBigOWith_of_eq_mul _ ((hφ c hpos).mono fun x => le_of_lt) huvφ
    /-
      🎉 no goals
    -/


alias ⟨IsLittleO.exists_eq_mul, _⟩ := isLittleO_iff_exists_eq_mul


theorem div_isBoundedUnder_of_isBigO {α : Type*} {l : Filter α} {f g : α → 𝕜} (h : f =O[l] g) :
    IsBoundedUnder (· ≤ ·) l fun x => ‖f x / g x‖ := by
  /-
    𝕜 : Type u_15
    inst✝ : NormedDivisionRing 𝕜
    α : Type u_17
    l : Filter α
    f g : α → 𝕜
    h : Asymptotics.IsBigO l f g
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l fun x => Norm.norm (HDiv. …
  -/
  obtain ⟨c, h₀, hc⟩ := h.exists_nonneg
  /-
    case intro.intro
    𝕜 : Type u_15
    inst✝ : NormedDivisionRing 𝕜
    α : Type u_17
    l : Filter α
    f g : α → 𝕜
    h : Asymptotics.IsBigO l f g
    c : Real
    h₀ : GE.ge c 0
    hc : Asymptotics.IsBigOWith c l f g
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l fun x => Norm.norm (HDiv. …
  -/
  refine ⟨c, eventually_map.2 (hc.bound.mono fun x hx => ?_)⟩
  /-
    case intro.intro
    𝕜 : Type u_15
    inst✝ : NormedDivisionRing 𝕜
    α : Type u_17
    l : Filter α
    f g : α → 𝕜
    h : Asymptotics.IsBigO l f g
    c : Real
    h₀ : GE.ge c 0
    hc : Asymptotics.IsBigOWith c l f g
    x : α
    hx : LE.le (Norm.norm (f x)) (HMul.hMul c (Norm.norm (g x)))
    ⊢ (fun x1 x2 => LE.le x1 x2) (Norm.norm (HDiv.hDiv (f x) (g x))) c
  -/
  rw [norm_div]
  /-
    case intro.intro
    𝕜 : Type u_15
    inst✝ : NormedDivisionRing 𝕜
    α : Type u_17
    l : Filter α
    f g : α → 𝕜
    h : Asymptotics.IsBigO l f g
    c : Real
    h₀ : GE.ge c 0
    hc : Asymptotics.IsBigOWith c l f g
    x : α
    hx : LE.le (Norm.norm (f x)) (HMul.hMul c (Norm.norm (g x)))
    ⊢ (fun x1 x2 => LE.le x1 x2) (HDiv.hDiv (Norm.norm (f x)) (Norm.norm (g x))) c
  -/
  exact div_le_of_le_mul₀ (norm_nonneg _) h₀ hx
  /-
    🎉 no goals
  -/


theorem isBigO_iff_div_isBoundedUnder {α : Type*} {l : Filter α} {f g : α → 𝕜}
    (hgf : ∀ᶠ x in l, g x = 0 → f x = 0) :
    f =O[l] g ↔ IsBoundedUnder (· ≤ ·) l fun x => ‖f x / g x‖ := by
  /-
    𝕜 : Type u_15
    inst✝ : NormedDivisionRing 𝕜
    α : Type u_17
    l : Filter α
    f g : α → 𝕜
    hgf : Filter.Eventually (fun x => Eq (g x) 0 → Eq (f x) 0) l
    ⊢ Iff (Asymptotics.IsBigO l f g) (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 …
  -/
  refine ⟨div_isBoundedUnder_of_isBigO, fun h => ?_⟩
  /-
    𝕜 : Type u_15
    inst✝ : NormedDivisionRing 𝕜
    α : Type u_17
    l : Filter α
    f g : α → 𝕜
    hgf : Filter.Eventually (fun x => Eq (g x) 0 → Eq (f x) 0) l
    h : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l fun x => Norm.norm (HDi …
    ⊢ Asymptotics.IsBigO l f g
  -/
  obtain ⟨c, hc⟩ := h
  /-
    case intro
    𝕜 : Type u_15
    inst✝ : NormedDivisionRing 𝕜
    α : Type u_17
    l : Filter α
    f g : α → 𝕜
    hgf : Filter.Eventually (fun x => Eq (g x) 0 → Eq (f x) 0) l
    c : Real
    hc : Filter.Eventually (fun x => (fun x1 x2 => LE.le x1 x2) x c) (Filter.map ( …
    ⊢ Asymptotics.IsBigO l f g
  -/
  simp only [eventually_map, norm_div] at hc
  /-
    case intro
    𝕜 : Type u_15
    inst✝ : NormedDivisionRing 𝕜
    α : Type u_17
    l : Filter α
    f g : α → 𝕜
    hgf : Filter.Eventually (fun x => Eq (g x) 0 → Eq (f x) 0) l
    c : Real
    hc : Filter.Eventually (fun a => LE.le (HDiv.hDiv (Norm.norm (f a)) (Norm.norm …
    ⊢ Asymptotics.IsBigO l f g
  -/
  refine IsBigO.of_bound c (hc.mp <| hgf.mono fun x hx₁ hx₂ => ?_)
  /-
    case intro
    𝕜 : Type u_15
    inst✝ : NormedDivisionRing 𝕜
    α : Type u_17
    l : Filter α
    f g : α → 𝕜
    hgf : Filter.Eventually (fun x => Eq (g x) 0 → Eq (f x) 0) l
    c : Real
    hc : Filter.Eventually (fun a => LE.le (HDiv.hDiv (Norm.norm (f a)) (Norm.norm …
    x : α
    hx₁ : Eq (g x) 0 → Eq (f x) 0
    hx₂ : LE.le (HDiv.hDiv (Norm.norm (f x)) (Norm.norm (g x))) c
    ⊢ LE.le (Norm.norm (f x)) (HMul.hMul c (Norm.norm (g x)))
  -/
  by_cases hgx : g x = 0
    /-
      case pos
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      α : Type u_17
      l : Filter α
      f g : α → 𝕜
      hgf : Filter.Eventually (fun x => Eq (g x) 0 → Eq (f x) 0) l
      c : Real
      hc : Filter.Eventually (fun a => LE.le (HDiv.hDiv (Norm.norm (f a)) (Norm.norm …
      x : α
      hx₁ : Eq (g x) 0 → Eq (f x) 0
      hx₂ : LE.le (HDiv.hDiv (Norm.norm (f x)) (Norm.norm (g x))) c
      hgx : Eq (g x) 0
      ⊢ LE.le (Norm.norm (f x)) (HMul.hMul c (Norm.norm (g x)))
    -/
  · simp [hx₁ hgx, hgx]
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      α : Type u_17
      l : Filter α
      f g : α → 𝕜
      hgf : Filter.Eventually (fun x => Eq (g x) 0 → Eq (f x) 0) l
      c : Real
      hc : Filter.Eventually (fun a => LE.le (HDiv.hDiv (Norm.norm (f a)) (Norm.norm …
      x : α
      hx₁ : Eq (g x) 0 → Eq (f x) 0
      hx₂ : LE.le (HDiv.hDiv (Norm.norm (f x)) (Norm.norm (g x))) c
      hgx : Not (Eq (g x) 0)
      ⊢ LE.le (Norm.norm (f x)) (HMul.hMul c (Norm.norm (g x)))
    -/
  · exact (div_le_iff₀ (norm_pos_iff.2 hgx)).mp hx₂
    /-
      🎉 no goals
    -/


theorem isBigO_of_div_tendsto_nhds {α : Type*} {l : Filter α} {f g : α → 𝕜}
    (hgf : ∀ᶠ x in l, g x = 0 → f x = 0) (c : 𝕜) (H : Filter.Tendsto (f / g) l (𝓝 c)) :
    f =O[l] g :=
  (isBigO_iff_div_isBoundedUnder hgf).2 <| H.norm.isBoundedUnder_le


theorem IsLittleO.tendsto_zero_of_tendsto {α E 𝕜 : Type*} [NormedAddCommGroup E] [NormedField 𝕜]
    {u : α → E} {v : α → 𝕜} {l : Filter α} {y : 𝕜} (huv : u =o[l] v) (hv : Tendsto v l (𝓝 y)) :
    Tendsto u l (𝓝 0) := by
  suffices h : u =o[l] fun _x => (1 : 𝕜) by
    rwa [isLittleO_one_iff] at h
  /-
    α : Type u_17
    E : Type u_18
    𝕜 : Type u_19
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedField 𝕜
    u : α → E
    v : α → 𝕜
    l : Filter α
    y : 𝕜
    huv : Asymptotics.IsLittleO l u v
    hv : Filter.Tendsto v l (nhds y)
    ⊢ Asymptotics.IsLittleO l u fun _x => 1
  -/
  exact huv.trans_isBigO (hv.isBigO_one 𝕜)
  /-
    🎉 no goals
  -/


theorem isLittleO_pow_pow {m n : ℕ} (h : m < n) : (fun x : 𝕜 => x ^ n) =o[𝓝 0] fun x => x ^ m := by
  /-
    𝕜 : Type u_15
    inst✝ : NormedDivisionRing 𝕜
    m n : Nat
    h : LT.lt m n
    ⊢ Asymptotics.IsLittleO (nhds 0) (fun x => HPow.hPow x n) fun x => HPow.hPow x m
  -/
  rcases lt_iff_exists_add.1 h with ⟨p, hp0 : 0 < p, rfl⟩
  suffices (fun x : 𝕜 => x ^ m * x ^ p) =o[𝓝 0] fun x => x ^ m * 1 ^ p by
    simpa only [pow_add, one_pow, mul_one]
  exact IsBigO.mul_isLittleO (isBigO_refl _ _)
    (IsLittleO.pow ((isLittleO_one_iff _).2 tendsto_id) hp0)


theorem isLittleO_norm_pow_norm_pow {m n : ℕ} (h : m < n) :
    (fun x : E' => ‖x‖ ^ n) =o[𝓝 0] fun x => ‖x‖ ^ m :=
  (isLittleO_pow_pow h).comp_tendsto tendsto_norm_zero


theorem isLittleO_pow_id {n : ℕ} (h : 1 < n) : (fun x : 𝕜 => x ^ n) =o[𝓝 0] fun x => x := by
  /-
    𝕜 : Type u_15
    inst✝ : NormedDivisionRing 𝕜
    n : Nat
    h : LT.lt 1 n
    ⊢ Asymptotics.IsLittleO (nhds 0) (fun x => HPow.hPow x n) fun x => x
  -/
  convert isLittleO_pow_pow h (𝕜 := 𝕜)
  /-
    case h.e'_8.h
    𝕜 : Type u_15
    inst✝ : NormedDivisionRing 𝕜
    n : Nat
    h : LT.lt 1 n
    x✝ : 𝕜
    ⊢ Eq x✝ (HPow.hPow x✝ 1)
  -/
  simp only [pow_one]
  /-
    🎉 no goals
  -/


theorem isLittleO_norm_pow_id {n : ℕ} (h : 1 < n) :
    (fun x : E' => ‖x‖ ^ n) =o[𝓝 0] fun x => x := by
  /-
    E' : Type u_6
    inst✝ : SeminormedAddCommGroup E'
    n : Nat
    h : LT.lt 1 n
    ⊢ Asymptotics.IsLittleO (nhds 0) (fun x => HPow.hPow (Norm.norm x) n) fun x => x
  -/
  have := @isLittleO_norm_pow_norm_pow E' _ _ _ h
  /-
    E' : Type u_6
    inst✝ : SeminormedAddCommGroup E'
    n : Nat
    h : LT.lt 1 n
    this : Asymptotics.IsLittleO (nhds 0) (fun x => HPow.hPow (Norm.norm x) n) fun …
    ⊢ Asymptotics.IsLittleO (nhds 0) (fun x => HPow.hPow (Norm.norm x) n) fun x => x
  -/
  simp only [pow_one] at this
  /-
    E' : Type u_6
    inst✝ : SeminormedAddCommGroup E'
    n : Nat
    h : LT.lt 1 n
    this : Asymptotics.IsLittleO (nhds 0) (fun x => HPow.hPow (Norm.norm x) n) fun …
    ⊢ Asymptotics.IsLittleO (nhds 0) (fun x => HPow.hPow (Norm.norm x) n) fun x => x
  -/
  exact isLittleO_norm_right.mp this
  /-
    🎉 no goals
  -/


theorem IsBigO.eq_zero_of_norm_pow_within {f : E'' → F''} {s : Set E''} {x₀ : E''} {n : ℕ}
    (h : f =O[𝓝[s] x₀] fun x => ‖x - x₀‖ ^ n) (hx₀ : x₀ ∈ s) (hn : n ≠ 0) : f x₀ = 0 :=
                                                /-
                                                  E'' : Type u_9
                                                  F'' : Type u_10
                                                  inst✝¹ : NormedAddCommGroup E''
                                                  inst✝ : NormedAddCommGroup F''
                                                  f : E'' → F''
                                                  s : Set E''
                                                  x₀ : E''
                                                  n : Nat
                                                  h : Asymptotics.IsBigO (nhdsWithin x₀ s) f fun x => HPow.hPow (Norm.norm (HSub …
                                                  hx₀ : Membership.mem s x₀
                                                  hn : Ne n 0
                                                  ⊢ Eq (HPow.hPow (Norm.norm (HSub.hSub x₀ x₀)) n) 0
                                                -/
  mem_of_mem_nhdsWithin hx₀ h.eq_zero_imp <| by simp_rw [sub_self, norm_zero, zero_pow hn]
                                                /-
                                                  🎉 no goals
                                                -/


theorem IsBigO.eq_zero_of_norm_pow {f : E'' → F''} {x₀ : E''} {n : ℕ}
    (h : f =O[𝓝 x₀] fun x => ‖x - x₀‖ ^ n) (hn : n ≠ 0) : f x₀ = 0 := by
  /-
    E'' : Type u_9
    F'' : Type u_10
    inst✝¹ : NormedAddCommGroup E''
    inst✝ : NormedAddCommGroup F''
    f : E'' → F''
    x₀ : E''
    n : Nat
    h : Asymptotics.IsBigO (nhds x₀) f fun x => HPow.hPow (Norm.norm (HSub.hSub x  …
    hn : Ne n 0
    ⊢ Eq (f x₀) 0
  -/
  rw [← nhdsWithin_univ] at h
  /-
    E'' : Type u_9
    F'' : Type u_10
    inst✝¹ : NormedAddCommGroup E''
    inst✝ : NormedAddCommGroup F''
    f : E'' → F''
    x₀ : E''
    n : Nat
    h : Asymptotics.IsBigO (nhdsWithin x₀ Set.univ) f fun x => HPow.hPow (Norm.nor …
    hn : Ne n 0
    ⊢ Eq (f x₀) 0
  -/
  exact h.eq_zero_of_norm_pow_within (mem_univ _) hn
  /-
    🎉 no goals
  -/


theorem isLittleO_pow_sub_pow_sub (x₀ : E') {n m : ℕ} (h : n < m) :
    (fun x => ‖x - x₀‖ ^ m) =o[𝓝 x₀] fun x => ‖x - x₀‖ ^ n :=
  haveI : Tendsto (fun x => ‖x - x₀‖) (𝓝 x₀) (𝓝 0) := by
    /-
      E' : Type u_6
      inst✝ : SeminormedAddCommGroup E'
      x₀ : E'
      n m : Nat
      h : LT.lt n m
      ⊢ Filter.Tendsto (fun x => Norm.norm (HSub.hSub x x₀)) (nhds x₀) (nhds 0)
    -/
    apply tendsto_norm_zero.comp
    /-
      E' : Type u_6
      inst✝ : SeminormedAddCommGroup E'
      x₀ : E'
      n m : Nat
      h : LT.lt n m
      ⊢ Filter.Tendsto (fun x => HSub.hSub x x₀) (nhds x₀) (nhds 0)
    -/
    rw [← sub_self x₀]
    /-
      E' : Type u_6
      inst✝ : SeminormedAddCommGroup E'
      x₀ : E'
      n m : Nat
      h : LT.lt n m
      ⊢ Filter.Tendsto (fun x => HSub.hSub x x₀) (nhds x₀) (nhds (HSub.hSub x₀ x₀))
    -/
    exact tendsto_id.sub tendsto_const_nhds
    /-
      🎉 no goals
    -/
  (isLittleO_pow_pow h).comp_tendsto this


theorem isLittleO_pow_sub_sub (x₀ : E') {m : ℕ} (h : 1 < m) :
    (fun x => ‖x - x₀‖ ^ m) =o[𝓝 x₀] fun x => x - x₀ := by
  /-
    E' : Type u_6
    inst✝ : SeminormedAddCommGroup E'
    x₀ : E'
    m : Nat
    h : LT.lt 1 m
    ⊢ Asymptotics.IsLittleO (nhds x₀) (fun x => HPow.hPow (Norm.norm (HSub.hSub x  …
  -/
  simpa only [isLittleO_norm_right, pow_one] using isLittleO_pow_sub_pow_sub x₀ h
  /-
    🎉 no goals
  -/


theorem IsBigOWith.right_le_sub_of_lt_one {f₁ f₂ : α → E'} (h : IsBigOWith c l f₁ f₂) (hc : c < 1) :
    IsBigOWith (1 / (1 - c)) l f₂ fun x => f₂ x - f₁ x :=
  IsBigOWith.of_bound <|
    mem_of_superset h.bound fun x hx => by
      /-
        α : Type u_1
        E' : Type u_6
        inst✝ : SeminormedAddCommGroup E'
        c : Real
        l : Filter α
        f₁ f₂ : α → E'
        h : Asymptotics.IsBigOWith c l f₁ f₂
        hc : LT.lt c 1
        x : α
        hx : Membership.mem (setOf fun x => (fun x => LE.le (Norm.norm (f₁ x)) (HMul.h …
        ⊢ Membership.mem (setOf fun x => (fun x => LE.le (Norm.norm (f₂ x)) (HMul.hMul …
      -/
      simp only [mem_setOf_eq] at hx ⊢
      /-
        α : Type u_1
        E' : Type u_6
        inst✝ : SeminormedAddCommGroup E'
        c : Real
        l : Filter α
        f₁ f₂ : α → E'
        h : Asymptotics.IsBigOWith c l f₁ f₂
        hc : LT.lt c 1
        x : α
        hx : LE.le (Norm.norm (f₁ x)) (HMul.hMul c (Norm.norm (f₂ x)))
        ⊢ LE.le (Norm.norm (f₂ x)) (HMul.hMul (HDiv.hDiv 1 (HSub.hSub 1 c)) (Norm.norm …
      -/
      rw [mul_comm, one_div, ← div_eq_mul_inv, le_div_iff₀, mul_sub, mul_one, mul_comm]
        /-
          α : Type u_1
          E' : Type u_6
          inst✝ : SeminormedAddCommGroup E'
          c : Real
          l : Filter α
          f₁ f₂ : α → E'
          h : Asymptotics.IsBigOWith c l f₁ f₂
          hc : LT.lt c 1
          x : α
          hx : LE.le (Norm.norm (f₁ x)) (HMul.hMul c (Norm.norm (f₂ x)))
          ⊢ LE.le (HSub.hSub (Norm.norm (f₂ x)) (HMul.hMul c (Norm.norm (f₂ x)))) (Norm. …
        -/
      · exact le_trans (sub_le_sub_left hx _) (norm_sub_norm_le _ _)
        /-
          🎉 no goals
        -/
        /-
          α : Type u_1
          E' : Type u_6
          inst✝ : SeminormedAddCommGroup E'
          c : Real
          l : Filter α
          f₁ f₂ : α → E'
          h : Asymptotics.IsBigOWith c l f₁ f₂
          hc : LT.lt c 1
          x : α
          hx : LE.le (Norm.norm (f₁ x)) (HMul.hMul c (Norm.norm (f₂ x)))
          ⊢ LT.lt 0 (HSub.hSub 1 c)
        -/
      · exact sub_pos.2 hc
        /-
          🎉 no goals
        -/


theorem IsBigOWith.right_le_add_of_lt_one {f₁ f₂ : α → E'} (h : IsBigOWith c l f₁ f₂) (hc : c < 1) :
    IsBigOWith (1 / (1 - c)) l f₂ fun x => f₁ x + f₂ x :=
  (h.neg_right.right_le_sub_of_lt_one hc).neg_right.of_neg_left.congr rfl (fun _ ↦ rfl) fun x ↦ by
    /-
      α : Type u_1
      E' : Type u_6
      inst✝ : SeminormedAddCommGroup E'
      c : Real
      l : Filter α
      f₁ f₂ : α → E'
      h : Asymptotics.IsBigOWith c l f₁ f₂
      hc : LT.lt c 1
      x : α
      ⊢ Eq (Neg.neg (HSub.hSub (Neg.neg (f₂ x)) (f₁ x))) (HAdd.hAdd (f₁ x) (f₂ x))
    -/
    rw [neg_sub, sub_neg_eq_add]
    /-
      🎉 no goals
    -/


theorem IsLittleO.right_isBigO_sub {f₁ f₂ : α → E'} (h : f₁ =o[l] f₂) :
    f₂ =O[l] fun x => f₂ x - f₁ x :=
  ((h.def' one_half_pos).right_le_sub_of_lt_one one_half_lt_one).isBigO


theorem IsLittleO.right_isBigO_add {f₁ f₂ : α → E'} (h : f₁ =o[l] f₂) :
    f₂ =O[l] fun x => f₁ x + f₂ x :=
  ((h.def' one_half_pos).right_le_add_of_lt_one one_half_lt_one).isBigO


theorem IsLittleO.right_isBigO_add' {f₁ f₂ : α → E'} (h : f₁ =o[l] f₂) :
    f₂ =O[l] (f₂ + f₁) :=
  add_comm f₁ f₂ ▸ h.right_isBigO_add


/-- If `f x = O(g x)` along `cofinite`, then there exists a positive constant `C` such that
`‖f x‖ ≤ C * ‖g x‖` whenever `g x ≠ 0`. -/
theorem bound_of_isBigO_cofinite (h : f =O[cofinite] g'') :
    ∃ C > 0, ∀ ⦃x⦄, g'' x ≠ 0 → ‖f x‖ ≤ C * ‖g'' x‖ := by
  /-
    α : Type u_1
    E : Type u_3
    F'' : Type u_10
    inst✝¹ : Norm E
    inst✝ : NormedAddCommGroup F''
    f : α → E
    g'' : α → F''
    h : Asymptotics.IsBigO Filter.cofinite f g''
    ⊢ Exists fun C => And (GT.gt C 0) (∀ ⦃x : α⦄, Ne (g'' x) 0 → LE.le (Norm.norm  …
  -/
  rcases h.exists_pos with ⟨C, C₀, hC⟩
  /-
    case intro.intro
    α : Type u_1
    E : Type u_3
    F'' : Type u_10
    inst✝¹ : Norm E
    inst✝ : NormedAddCommGroup F''
    f : α → E
    g'' : α → F''
    h : Asymptotics.IsBigO Filter.cofinite f g''
    C : Real
    C₀ : GT.gt C 0
    hC : Asymptotics.IsBigOWith C Filter.cofinite f g''
    ⊢ Exists fun C => And (GT.gt C 0) (∀ ⦃x : α⦄, Ne (g'' x) 0 → LE.le (Norm.norm  …
  -/
  rw [IsBigOWith_def, eventually_cofinite] at hC
  /-
    case intro.intro
    α : Type u_1
    E : Type u_3
    F'' : Type u_10
    inst✝¹ : Norm E
    inst✝ : NormedAddCommGroup F''
    f : α → E
    g'' : α → F''
    h : Asymptotics.IsBigO Filter.cofinite f g''
    C : Real
    C₀ : GT.gt C 0
    hC : (setOf fun x => Not (LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.norm (g'' …
    ⊢ Exists fun C => And (GT.gt C 0) (∀ ⦃x : α⦄, Ne (g'' x) 0 → LE.le (Norm.norm  …
  -/
  rcases (hC.toFinset.image fun x => ‖f x‖ / ‖g'' x‖).exists_le with ⟨C', hC'⟩
  /-
    case intro.intro.intro
    α : Type u_1
    E : Type u_3
    F'' : Type u_10
    inst✝¹ : Norm E
    inst✝ : NormedAddCommGroup F''
    f : α → E
    g'' : α → F''
    h : Asymptotics.IsBigO Filter.cofinite f g''
    C : Real
    C₀ : GT.gt C 0
    hC : (setOf fun x => Not (LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.norm (g'' …
    C' : Real
    hC' : ∀ (i : Real), Membership.mem (Finset.image (fun x => HDiv.hDiv (Norm.nor …
    ⊢ Exists fun C => And (GT.gt C 0) (∀ ⦃x : α⦄, Ne (g'' x) 0 → LE.le (Norm.norm  …
  -/
  have : ∀ x, C * ‖g'' x‖ < ‖f x‖ → ‖f x‖ / ‖g'' x‖ ≤ C' := by simpa using hC'
  /-
    case intro.intro.intro
    α : Type u_1
    E : Type u_3
    F'' : Type u_10
    inst✝¹ : Norm E
    inst✝ : NormedAddCommGroup F''
    f : α → E
    g'' : α → F''
    h : Asymptotics.IsBigO Filter.cofinite f g''
    C : Real
    C₀ : GT.gt C 0
    hC : (setOf fun x => Not (LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.norm (g'' …
    C' : Real
    hC' : ∀ (i : Real), Membership.mem (Finset.image (fun x => HDiv.hDiv (Norm.nor …
    this : ∀ (x : α), LT.lt (HMul.hMul C (Norm.norm (g'' x))) (Norm.norm (f x)) →  …
    ⊢ Exists fun C => And (GT.gt C 0) (∀ ⦃x : α⦄, Ne (g'' x) 0 → LE.le (Norm.norm  …
  -/
  refine ⟨max C C', lt_max_iff.2 (Or.inl C₀), fun x h₀ => ?_⟩
  /-
    case intro.intro.intro
    α : Type u_1
    E : Type u_3
    F'' : Type u_10
    inst✝¹ : Norm E
    inst✝ : NormedAddCommGroup F''
    f : α → E
    g'' : α → F''
    h : Asymptotics.IsBigO Filter.cofinite f g''
    C : Real
    C₀ : GT.gt C 0
    hC : (setOf fun x => Not (LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.norm (g'' …
    C' : Real
    hC' : ∀ (i : Real), Membership.mem (Finset.image (fun x => HDiv.hDiv (Norm.nor …
    this : ∀ (x : α), LT.lt (HMul.hMul C (Norm.norm (g'' x))) (Norm.norm (f x)) →  …
    x : α
    h₀ : Ne (g'' x) 0
    ⊢ LE.le (Norm.norm (f x)) (HMul.hMul (Max.max C C') (Norm.norm (g'' x)))
  -/
  rw [max_mul_of_nonneg _ _ (norm_nonneg _), le_max_iff, or_iff_not_imp_left, not_le]
  /-
    case intro.intro.intro
    α : Type u_1
    E : Type u_3
    F'' : Type u_10
    inst✝¹ : Norm E
    inst✝ : NormedAddCommGroup F''
    f : α → E
    g'' : α → F''
    h : Asymptotics.IsBigO Filter.cofinite f g''
    C : Real
    C₀ : GT.gt C 0
    hC : (setOf fun x => Not (LE.le (Norm.norm (f x)) (HMul.hMul C (Norm.norm (g'' …
    C' : Real
    hC' : ∀ (i : Real), Membership.mem (Finset.image (fun x => HDiv.hDiv (Norm.nor …
    this : ∀ (x : α), LT.lt (HMul.hMul C (Norm.norm (g'' x))) (Norm.norm (f x)) →  …
    x : α
    h₀ : Ne (g'' x) 0
    ⊢ LT.lt (HMul.hMul C (Norm.norm (g'' x))) (Norm.norm (f x)) → LE.le (Norm.norm …
  -/
  exact fun hx => (div_le_iff₀ (norm_pos_iff.2 h₀)).1 (this _ hx)
  /-
    🎉 no goals
  -/


theorem isBigO_cofinite_iff (h : ∀ x, g'' x = 0 → f'' x = 0) :
    f'' =O[cofinite] g'' ↔ ∃ C, ∀ x, ‖f'' x‖ ≤ C * ‖g'' x‖ := by
  classical
  exact ⟨fun h' =>
    let ⟨C, _C₀, hC⟩ := bound_of_isBigO_cofinite h'
    ⟨C, fun x => if hx : g'' x = 0 then by simp [h _ hx, hx] else hC hx⟩,
    fun h => (isBigO_top.2 h).mono le_top⟩


theorem bound_of_isBigO_nat_atTop {f : ℕ → E} {g'' : ℕ → E''} (h : f =O[atTop] g'') :
    ∃ C > 0, ∀ ⦃x⦄, g'' x ≠ 0 → ‖f x‖ ≤ C * ‖g'' x‖ :=
                                 /-
                                   E : Type u_3
                                   E'' : Type u_9
                                   inst✝¹ : Norm E
                                   inst✝ : NormedAddCommGroup E''
                                   f : Nat → E
                                   g'' : Nat → E''
                                   h : Asymptotics.IsBigO Filter.atTop f g''
                                   ⊢ Asymptotics.IsBigO Filter.cofinite f g''
                                 -/
  bound_of_isBigO_cofinite <| by rwa [Nat.cofinite_eq_atTop]
                                 /-
                                   🎉 no goals
                                 -/


theorem isBigO_nat_atTop_iff {f : ℕ → E''} {g : ℕ → F''} (h : ∀ x, g x = 0 → f x = 0) :
    f =O[atTop] g ↔ ∃ C, ∀ x, ‖f x‖ ≤ C * ‖g x‖ := by
  /-
    E'' : Type u_9
    F'' : Type u_10
    inst✝¹ : NormedAddCommGroup E''
    inst✝ : NormedAddCommGroup F''
    f : Nat → E''
    g : Nat → F''
    h : ∀ (x : Nat), Eq (g x) 0 → Eq (f x) 0
    ⊢ Iff (Asymptotics.IsBigO Filter.atTop f g) (Exists fun C => ∀ (x : Nat), LE.l …
  -/
  rw [← Nat.cofinite_eq_atTop, isBigO_cofinite_iff h]
  /-
    🎉 no goals
  -/


theorem isBigO_one_nat_atTop_iff {f : ℕ → E''} :
    f =O[atTop] (fun _n => 1 : ℕ → ℝ) ↔ ∃ C, ∀ n, ‖f n‖ ≤ C :=
  Iff.trans (isBigO_nat_atTop_iff fun _ h => (one_ne_zero h).elim) <| by
    /-
      E'' : Type u_9
      inst✝ : NormedAddCommGroup E''
      f : Nat → E''
      ⊢ Iff (Exists fun C => ∀ (x : Nat), LE.le (Norm.norm (f x)) (HMul.hMul C (Norm …
    -/
    simp only [norm_one, mul_one]
    /-
      🎉 no goals
    -/


theorem isBigOWith_pi {ι : Type*} [Fintype ι] {E' : ι → Type*} [∀ i, NormedAddCommGroup (E' i)]
    {f : α → ∀ i, E' i} {C : ℝ} (hC : 0 ≤ C) :
    IsBigOWith C l f g' ↔ ∀ i, IsBigOWith C l (fun x => f x i) g' := by
  /-
    α : Type u_1
    F' : Type u_7
    inst✝² : SeminormedAddCommGroup F'
    g' : α → F'
    l : Filter α
    ι : Type u_17
    inst✝¹ : Fintype ι
    E' : ι → Type u_18
    inst✝ : (i : ι) → NormedAddCommGroup (E' i)
    f : α → (i : ι) → E' i
    C : Real
    hC : LE.le 0 C
    ⊢ Iff (Asymptotics.IsBigOWith C l f g') (∀ (i : ι), Asymptotics.IsBigOWith C l …
  -/
  have : ∀ x, 0 ≤ C * ‖g' x‖ := fun x => mul_nonneg hC (norm_nonneg _)
  /-
    α : Type u_1
    F' : Type u_7
    inst✝² : SeminormedAddCommGroup F'
    g' : α → F'
    l : Filter α
    ι : Type u_17
    inst✝¹ : Fintype ι
    E' : ι → Type u_18
    inst✝ : (i : ι) → NormedAddCommGroup (E' i)
    f : α → (i : ι) → E' i
    C : Real
    hC : LE.le 0 C
    this : ∀ (x : α), LE.le 0 (HMul.hMul C (Norm.norm (g' x)))
    ⊢ Iff (Asymptotics.IsBigOWith C l f g') (∀ (i : ι), Asymptotics.IsBigOWith C l …
  -/
  simp only [isBigOWith_iff, pi_norm_le_iff_of_nonneg (this _), eventually_all]
  /-
    🎉 no goals
  -/


@[simp]
theorem isBigO_pi {ι : Type*} [Fintype ι] {E' : ι → Type*} [∀ i, NormedAddCommGroup (E' i)]
    {f : α → ∀ i, E' i} : f =O[l] g' ↔ ∀ i, (fun x => f x i) =O[l] g' := by
  /-
    α : Type u_1
    F' : Type u_7
    inst✝² : SeminormedAddCommGroup F'
    g' : α → F'
    l : Filter α
    ι : Type u_17
    inst✝¹ : Fintype ι
    E' : ι → Type u_18
    inst✝ : (i : ι) → NormedAddCommGroup (E' i)
    f : α → (i : ι) → E' i
    ⊢ Iff (Asymptotics.IsBigO l f g') (∀ (i : ι), Asymptotics.IsBigO l (fun x => f …
  -/
  simp only [isBigO_iff_eventually_isBigOWith, ← eventually_all]
  /-
    α : Type u_1
    F' : Type u_7
    inst✝² : SeminormedAddCommGroup F'
    g' : α → F'
    l : Filter α
    ι : Type u_17
    inst✝¹ : Fintype ι
    E' : ι → Type u_18
    inst✝ : (i : ι) → NormedAddCommGroup (E' i)
    f : α → (i : ι) → E' i
    ⊢ Iff (Filter.Eventually (fun c => Asymptotics.IsBigOWith c l f g') Filter.atT …
  -/
  exact eventually_congr (eventually_atTop.2 ⟨0, fun c => isBigOWith_pi⟩)
  /-
    🎉 no goals
  -/


@[simp]
theorem isLittleO_pi {ι : Type*} [Fintype ι] {E' : ι → Type*} [∀ i, NormedAddCommGroup (E' i)]
    {f : α → ∀ i, E' i} : f =o[l] g' ↔ ∀ i, (fun x => f x i) =o[l] g' := by
  /-
    α : Type u_1
    F' : Type u_7
    inst✝² : SeminormedAddCommGroup F'
    g' : α → F'
    l : Filter α
    ι : Type u_17
    inst✝¹ : Fintype ι
    E' : ι → Type u_18
    inst✝ : (i : ι) → NormedAddCommGroup (E' i)
    f : α → (i : ι) → E' i
    ⊢ Iff (Asymptotics.IsLittleO l f g') (∀ (i : ι), Asymptotics.IsLittleO l (fun  …
  -/
  simp +contextual only [IsLittleO_def, isBigOWith_pi, le_of_lt]
  /-
    α : Type u_1
    F' : Type u_7
    inst✝² : SeminormedAddCommGroup F'
    g' : α → F'
    l : Filter α
    ι : Type u_17
    inst✝¹ : Fintype ι
    E' : ι → Type u_18
    inst✝ : (i : ι) → NormedAddCommGroup (E' i)
    f : α → (i : ι) → E' i
    ⊢ Iff (∀ ⦃c : Real⦄, LT.lt 0 c → ∀ (i : ι), Asymptotics.IsBigOWith c l (fun x  …
  -/
  exact ⟨fun h i c hc => h hc i, fun h c hc i => h i hc⟩
  /-
    🎉 no goals
  -/


theorem IsBigO.natCast_atTop {R : Type*} [StrictOrderedSemiring R] [Archimedean R]
    {f : R → E} {g : R → F} (h : f =O[atTop] g) :
    (fun (n : ℕ) => f n) =O[atTop] (fun n => g n) :=
  IsBigO.comp_tendsto h tendsto_natCast_atTop_atTop


@[deprecated (since := "2024-04-17")]
alias IsBigO.nat_cast_atTop := IsBigO.natCast_atTop


theorem IsLittleO.natCast_atTop {R : Type*} [StrictOrderedSemiring R] [Archimedean R]
    {f : R → E} {g : R → F} (h : f =o[atTop] g) :
    (fun (n : ℕ) => f n) =o[atTop] (fun n => g n) :=
  IsLittleO.comp_tendsto h tendsto_natCast_atTop_atTop


@[deprecated (since := "2024-04-17")]
alias IsLittleO.nat_cast_atTop := IsLittleO.natCast_atTop


theorem isBigO_atTop_iff_eventually_exists {α : Type*} [SemilatticeSup α] [Nonempty α]
    {f : α → E} {g : α → F} : f =O[atTop] g ↔ ∀ᶠ n₀ in atTop, ∃ c, ∀ n ≥ n₀, ‖f n‖ ≤ c * ‖g n‖ := by
  /-
    E : Type u_3
    F : Type u_4
    inst✝³ : Norm E
    inst✝² : Norm F
    α : Type u_17
    inst✝¹ : SemilatticeSup α
    inst✝ : Nonempty α
    f : α → E
    g : α → F
    ⊢ Iff (Asymptotics.IsBigO Filter.atTop f g) (Filter.Eventually (fun n₀ => Exis …
  -/
  rw [isBigO_iff, exists_eventually_atTop]
  /-
    🎉 no goals
  -/


theorem isBigO_atTop_iff_eventually_exists_pos {α : Type*}
    [SemilatticeSup α] [Nonempty α] {f : α → G} {g : α → G'} :
    f =O[atTop] g ↔ ∀ᶠ n₀ in atTop, ∃ c > 0, ∀ n ≥ n₀, c * ‖f n‖ ≤ ‖g n‖ := by
  /-
    G : Type u_5
    G' : Type u_8
    inst✝³ : Norm G
    inst✝² : SeminormedAddCommGroup G'
    α : Type u_17
    inst✝¹ : SemilatticeSup α
    inst✝ : Nonempty α
    f : α → G
    g : α → G'
    ⊢ Iff (Asymptotics.IsBigO Filter.atTop f g) (Filter.Eventually (fun n₀ => Exis …
  -/
  simp_rw [isBigO_iff'', ← exists_prop, Subtype.exists', exists_eventually_atTop]
  /-
    🎉 no goals
  -/


lemma isBigO_mul_iff_isBigO_div {f g h : α → 𝕜} (hf : ∀ᶠ x in l, f x ≠ 0) :
    (fun x ↦ f x * g x) =O[l] h ↔ g =O[l] (fun x ↦ h x / f x) := by
  /-
    α : Type u_1
    𝕜 : Type u_15
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    f g h : α → 𝕜
    hf : Filter.Eventually (fun x => Ne (f x) 0) l
    ⊢ Iff (Asymptotics.IsBigO l (fun x => HMul.hMul (f x) (g x)) h) (Asymptotics.I …
  -/
  rw [isBigO_iff', isBigO_iff']
  /-
    α : Type u_1
    𝕜 : Type u_15
    inst✝ : NormedDivisionRing 𝕜
    l : Filter α
    f g h : α → 𝕜
    hf : Filter.Eventually (fun x => Ne (f x) 0) l
    ⊢ Iff (Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun x => LE.le (Nor …
  -/
  refine ⟨fun ⟨c, hc, H⟩ ↦ ⟨c, hc, ?_⟩, fun ⟨c, hc, H⟩ ↦ ⟨c, hc, ?_⟩⟩ <;>
    /-
      case refine_1
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      l : Filter α
      f g h : α → 𝕜
      hf : Filter.Eventually (fun x => Ne (f x) 0) l
      x✝ : Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun x => LE.le (Norm. …
      c : Real
      hc : GT.gt c 0
      H : Filter.Eventually (fun x => LE.le (Norm.norm (HMul.hMul (f x) (g x))) (HMu …
      ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (g x)) (HMul.hMul c (Norm.norm  …
    -/
    /-
      case refine_1
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      l : Filter α
      f g h : α → 𝕜
      hf : Filter.Eventually (fun x => Ne (f x) 0) l
      x✝ : Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun x => LE.le (Norm. …
      c : Real
      hc : GT.gt c 0
      H : Filter.Eventually (fun x => LE.le (Norm.norm (HMul.hMul (f x) (g x))) (HMu …
      x : α
      hx : Ne (f x) 0
      ⊢ Iff (LE.le (Norm.norm (HMul.hMul (f x) (g x))) (HMul.hMul c (Norm.norm (h x) …
    -/
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      𝕜 : Type u_15
      inst✝ : NormedDivisionRing 𝕜
      l : Filter α
      f g h : α → 𝕜
      hf : Filter.Eventually (fun x => Ne (f x) 0) l
      x✝ : Exists fun c => And (GT.gt c 0) (Filter.Eventually (fun x => LE.le (Norm. …
      c : Real
      hc : GT.gt c 0
      H : Filter.Eventually (fun x => LE.le (Norm.norm (g x)) (HMul.hMul c (Norm.nor …
      x : α
      hx : Ne (f x) 0
      ⊢ Iff (LE.le (Norm.norm (g x)) (HMul.hMul c (Norm.norm (HDiv.hDiv (h x) (f x)) …
    -/
    rw [norm_mul, norm_div, ← mul_div_assoc, le_div_iff₀' (norm_pos_iff.mpr hx)]
    /-
      🎉 no goals
    -/


theorem summable_of_isBigO {ι E} [SeminormedAddCommGroup E] [CompleteSpace E]
    {f : ι → E} {g : ι → ℝ} (hg : Summable g) (h : f =O[cofinite] g) : Summable f :=
  let ⟨C, hC⟩ := h.isBigOWith
  .of_norm_bounded_eventually (fun x => C * ‖g x‖) (hg.abs.mul_left _) hC.bound


theorem summable_of_isBigO_nat {E} [SeminormedAddCommGroup E] [CompleteSpace E]
    {f : ℕ → E} {g : ℕ → ℝ} (hg : Summable g) (h : f =O[atTop] g) : Summable f :=
  summable_of_isBigO hg <| Nat.cofinite_eq_atTop.symm ▸ h


lemma Asymptotics.IsBigO.comp_summable_norm {ι E F : Type*}
    [SeminormedAddCommGroup E] [SeminormedAddCommGroup F] {f : E → F} {g : ι → E}
    (hf : f =O[𝓝 0] id) (hg : Summable (‖g ·‖)) : Summable (‖f <| g ·‖) :=
  summable_of_isBigO hg <| hf.norm_norm.comp_tendsto <|
    tendsto_zero_iff_norm_tendsto_zero.2 hg.tendsto_cofinite_zero


/-- Transfer `IsBigOWith` over a `PartialHomeomorph`. -/
theorem isBigOWith_congr (e : PartialHomeomorph α β) {b : β} (hb : b ∈ e.target) {f : β → E}
    {g : β → F} {C : ℝ} : IsBigOWith C (𝓝 b) f g ↔ IsBigOWith C (𝓝 (e.symm b)) (f ∘ e) (g ∘ e) :=
  ⟨fun h =>
    h.comp_tendsto <| by
      /-
        α : Type u_1
        β : Type u_2
        inst✝³ : TopologicalSpace α
        inst✝² : TopologicalSpace β
        E : Type u_3
        inst✝¹ : Norm E
        F : Type u_4
        inst✝ : Norm F
        e : PartialHomeomorph α β
        b : β
        hb : Membership.mem e.target b
        f : β → E
        g : β → F
        C : Real
        h : Asymptotics.IsBigOWith C (nhds b) f g
        ⊢ Filter.Tendsto (↑e) (nhds (↑e.symm b)) (nhds b)
      -/
      have := e.continuousAt (e.map_target hb)
      /-
        α : Type u_1
        β : Type u_2
        inst✝³ : TopologicalSpace α
        inst✝² : TopologicalSpace β
        E : Type u_3
        inst✝¹ : Norm E
        F : Type u_4
        inst✝ : Norm F
        e : PartialHomeomorph α β
        b : β
        hb : Membership.mem e.target b
        f : β → E
        g : β → F
        C : Real
        h : Asymptotics.IsBigOWith C (nhds b) f g
        this : ContinuousAt (↑e) (↑e.symm b)
        ⊢ Filter.Tendsto (↑e) (nhds (↑e.symm b)) (nhds b)
      -/
      rwa [ContinuousAt, e.rightInvOn hb] at this,
      /-
        🎉 no goals
      -/
    fun h =>
    (h.comp_tendsto (e.continuousAt_symm hb)).congr' rfl
      ((e.eventually_right_inverse hb).mono fun _ hx => congr_arg f hx)
      ((e.eventually_right_inverse hb).mono fun _ hx => congr_arg g hx)⟩


/-- Transfer `IsBigO` over a `PartialHomeomorph`. -/
theorem isBigO_congr (e : PartialHomeomorph α β) {b : β} (hb : b ∈ e.target) {f : β → E}
    {g : β → F} : f =O[𝓝 b] g ↔ (f ∘ e) =O[𝓝 (e.symm b)] (g ∘ e) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    E : Type u_3
    inst✝¹ : Norm E
    F : Type u_4
    inst✝ : Norm F
    e : PartialHomeomorph α β
    b : β
    hb : Membership.mem e.target b
    f : β → E
    g : β → F
    ⊢ Iff (Asymptotics.IsBigO (nhds b) f g) (Asymptotics.IsBigO (nhds (↑e.symm b)) …
  -/
  simp only [IsBigO_def]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    E : Type u_3
    inst✝¹ : Norm E
    F : Type u_4
    inst✝ : Norm F
    e : PartialHomeomorph α β
    b : β
    hb : Membership.mem e.target b
    f : β → E
    g : β → F
    ⊢ Iff (Exists fun c => Asymptotics.IsBigOWith c (nhds b) f g) (Exists fun c => …
  -/
  exact exists_congr fun C => e.isBigOWith_congr hb
  /-
    🎉 no goals
  -/


/-- Transfer `IsLittleO` over a `PartialHomeomorph`. -/
theorem isLittleO_congr (e : PartialHomeomorph α β) {b : β} (hb : b ∈ e.target) {f : β → E}
    {g : β → F} : f =o[𝓝 b] g ↔ (f ∘ e) =o[𝓝 (e.symm b)] (g ∘ e) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    E : Type u_3
    inst✝¹ : Norm E
    F : Type u_4
    inst✝ : Norm F
    e : PartialHomeomorph α β
    b : β
    hb : Membership.mem e.target b
    f : β → E
    g : β → F
    ⊢ Iff (Asymptotics.IsLittleO (nhds b) f g) (Asymptotics.IsLittleO (nhds (↑e.sy …
  -/
  simp only [IsLittleO_def]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    E : Type u_3
    inst✝¹ : Norm E
    F : Type u_4
    inst✝ : Norm F
    e : PartialHomeomorph α β
    b : β
    hb : Membership.mem e.target b
    f : β → E
    g : β → F
    ⊢ Iff (∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c (nhds b) f g) (∀ ⦃c  …
  -/
  exact forall₂_congr fun c _hc => e.isBigOWith_congr hb
  /-
    🎉 no goals
  -/


/-- Transfer `IsBigOWith` over a `Homeomorph`. -/
theorem isBigOWith_congr (e : α ≃ₜ β) {b : β} {f : β → E} {g : β → F} {C : ℝ} :
    IsBigOWith C (𝓝 b) f g ↔ IsBigOWith C (𝓝 (e.symm b)) (f ∘ e) (g ∘ e) :=
  e.toPartialHomeomorph.isBigOWith_congr trivial


/-- Transfer `IsBigO` over a `Homeomorph`. -/
theorem isBigO_congr (e : α ≃ₜ β) {b : β} {f : β → E} {g : β → F} :
    f =O[𝓝 b] g ↔ (f ∘ e) =O[𝓝 (e.symm b)] (g ∘ e) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    E : Type u_3
    inst✝¹ : Norm E
    F : Type u_4
    inst✝ : Norm F
    e : Homeomorph α β
    b : β
    f : β → E
    g : β → F
    ⊢ Iff (Asymptotics.IsBigO (nhds b) f g) (Asymptotics.IsBigO (nhds (e.symm b))  …
  -/
  simp only [IsBigO_def]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    E : Type u_3
    inst✝¹ : Norm E
    F : Type u_4
    inst✝ : Norm F
    e : Homeomorph α β
    b : β
    f : β → E
    g : β → F
    ⊢ Iff (Exists fun c => Asymptotics.IsBigOWith c (nhds b) f g) (Exists fun c => …
  -/
  exact exists_congr fun C => e.isBigOWith_congr
  /-
    🎉 no goals
  -/


/-- Transfer `IsLittleO` over a `Homeomorph`. -/
theorem isLittleO_congr (e : α ≃ₜ β) {b : β} {f : β → E} {g : β → F} :
    f =o[𝓝 b] g ↔ (f ∘ e) =o[𝓝 (e.symm b)] (g ∘ e) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    E : Type u_3
    inst✝¹ : Norm E
    F : Type u_4
    inst✝ : Norm F
    e : Homeomorph α β
    b : β
    f : β → E
    g : β → F
    ⊢ Iff (Asymptotics.IsLittleO (nhds b) f g) (Asymptotics.IsLittleO (nhds (e.sym …
  -/
  simp only [IsLittleO_def]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    E : Type u_3
    inst✝¹ : Norm E
    F : Type u_4
    inst✝ : Norm F
    e : Homeomorph α β
    b : β
    f : β → E
    g : β → F
    ⊢ Iff (∀ ⦃c : Real⦄, LT.lt 0 c → Asymptotics.IsBigOWith c (nhds b) f g) (∀ ⦃c  …
  -/
  exact forall₂_congr fun c _hc => e.isBigOWith_congr
  /-
    🎉 no goals
  -/


protected theorem isBigOWith_principal
    (hf : ContinuousOn f s) (hs : IsCompact s) (hc : ‖c‖ ≠ 0) :
    IsBigOWith (sSup (Norm.norm '' (f '' s)) / ‖c‖) (𝓟 s) f fun _ => c := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : TopologicalSpace α
    s : Set α
    f : α → E
    c : F
    inst✝¹ : SeminormedAddGroup E
    inst✝ : Norm F
    hf : ContinuousOn f s
    hs : IsCompact s
    hc : Ne (Norm.norm c) 0
    ⊢ Asymptotics.IsBigOWith (HDiv.hDiv (SupSet.sSup (Set.image Norm.norm (Set.ima …
  -/
  rw [isBigOWith_principal, div_mul_cancel₀ _ hc]
  exact fun x hx ↦ hs.image_of_continuousOn hf |>.image continuous_norm
   |>.isLUB_sSup (Set.image_nonempty.mpr <| Set.image_nonempty.mpr ⟨x, hx⟩)
   |>.left <| Set.mem_image_of_mem _ <| Set.mem_image_of_mem _ hx


protected theorem isBigO_principal (hf : ContinuousOn f s) (hs : IsCompact s)
    (hc : ‖c‖ ≠ 0) : f =O[𝓟 s] fun _ => c :=
  (hf.isBigOWith_principal hs hc).isBigO


protected theorem isBigOWith_rev_principal
    (hf : ContinuousOn f s) (hs : IsCompact s) (hC : ∀ i ∈ s, f i ≠ 0) (c : F) :
    IsBigOWith (‖c‖ / sInf (Norm.norm '' (f '' s))) (𝓟 s) (fun _ => c) f := by
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : TopologicalSpace α
    s : Set α
    f : α → E
    inst✝¹ : NormedAddGroup E
    inst✝ : SeminormedAddGroup F
    hf : ContinuousOn f s
    hs : IsCompact s
    hC : ∀ (i : α), Membership.mem s i → Ne (f i) 0
    c : F
    ⊢ Asymptotics.IsBigOWith (HDiv.hDiv (Norm.norm c) (InfSet.sInf (Set.image Norm …
  -/
  refine isBigOWith_principal.mpr fun x hx ↦ ?_
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : TopologicalSpace α
    s : Set α
    f : α → E
    inst✝¹ : NormedAddGroup E
    inst✝ : SeminormedAddGroup F
    hf : ContinuousOn f s
    hs : IsCompact s
    hC : ∀ (i : α), Membership.mem s i → Ne (f i) 0
    c : F
    x : α
    hx : Membership.mem s x
    ⊢ LE.le (Norm.norm c) (HMul.hMul (HDiv.hDiv (Norm.norm c) (InfSet.sInf (Set.im …
  -/
  rw [mul_comm_div]
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : TopologicalSpace α
    s : Set α
    f : α → E
    inst✝¹ : NormedAddGroup E
    inst✝ : SeminormedAddGroup F
    hf : ContinuousOn f s
    hs : IsCompact s
    hC : ∀ (i : α), Membership.mem s i → Ne (f i) 0
    c : F
    x : α
    hx : Membership.mem s x
    ⊢ LE.le (Norm.norm c) (HMul.hMul (Norm.norm c) (HDiv.hDiv (Norm.norm (f x)) (I …
  -/
  replace hs := hs.image_of_continuousOn hf |>.image continuous_norm
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : TopologicalSpace α
    s : Set α
    f : α → E
    inst✝¹ : NormedAddGroup E
    inst✝ : SeminormedAddGroup F
    hf : ContinuousOn f s
    hC : ∀ (i : α), Membership.mem s i → Ne (f i) 0
    c : F
    x : α
    hx : Membership.mem s x
    hs : IsCompact (Set.image (fun a => Norm.norm a) (Set.image f s))
    ⊢ LE.le (Norm.norm c) (HMul.hMul (Norm.norm c) (HDiv.hDiv (Norm.norm (f x)) (I …
  -/
  have h_sInf := hs.isGLB_sInf <| Set.image_nonempty.mpr <| Set.image_nonempty.mpr ⟨x, hx⟩
  refine le_mul_of_one_le_right (norm_nonneg c) <| (one_le_div ?_).mpr <|
    h_sInf.1 <| Set.mem_image_of_mem _ <| Set.mem_image_of_mem _ hx
  /-
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : TopologicalSpace α
    s : Set α
    f : α → E
    inst✝¹ : NormedAddGroup E
    inst✝ : SeminormedAddGroup F
    hf : ContinuousOn f s
    hC : ∀ (i : α), Membership.mem s i → Ne (f i) 0
    c : F
    x : α
    hx : Membership.mem s x
    hs : IsCompact (Set.image (fun a => Norm.norm a) (Set.image f s))
    h_sInf : IsGLB (Set.image (fun a => Norm.norm a) (Set.image f s)) (InfSet.sInf …
    ⊢ LT.lt 0 (InfSet.sInf (Set.image Norm.norm (Set.image f s)))
  -/
  obtain ⟨_, ⟨x, hx, hCx⟩, hnormCx⟩ := hs.sInf_mem h_sInf.nonempty
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : TopologicalSpace α
    s : Set α
    f : α → E
    inst✝¹ : NormedAddGroup E
    inst✝ : SeminormedAddGroup F
    hf : ContinuousOn f s
    hC : ∀ (i : α), Membership.mem s i → Ne (f i) 0
    c : F
    x✝ : α
    hx✝ : Membership.mem s x✝
    hs : IsCompact (Set.image (fun a => Norm.norm a) (Set.image f s))
    h_sInf : IsGLB (Set.image (fun a => Norm.norm a) (Set.image f s)) (InfSet.sInf …
    w✝ : E
    hnormCx : Eq ((fun a => Norm.norm a) w✝) (InfSet.sInf (Set.image (fun a => Nor …
    x : α
    hx : Membership.mem s x
    hCx : Eq (f x) w✝
    ⊢ LT.lt 0 (InfSet.sInf (Set.image Norm.norm (Set.image f s)))
  -/
  rw [← hnormCx, ← hCx]
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝² : TopologicalSpace α
    s : Set α
    f : α → E
    inst✝¹ : NormedAddGroup E
    inst✝ : SeminormedAddGroup F
    hf : ContinuousOn f s
    hC : ∀ (i : α), Membership.mem s i → Ne (f i) 0
    c : F
    x✝ : α
    hx✝ : Membership.mem s x✝
    hs : IsCompact (Set.image (fun a => Norm.norm a) (Set.image f s))
    h_sInf : IsGLB (Set.image (fun a => Norm.norm a) (Set.image f s)) (InfSet.sInf …
    w✝ : E
    hnormCx : Eq ((fun a => Norm.norm a) w✝) (InfSet.sInf (Set.image (fun a => Nor …
    x : α
    hx : Membership.mem s x
    hCx : Eq (f x) w✝
    ⊢ LT.lt 0 ((fun a => Norm.norm a) (f x))
  -/
  exact (norm_ne_zero_iff.mpr (hC x hx)).symm.lt_of_le (norm_nonneg _)
  /-
    🎉 no goals
  -/


protected theorem isBigO_rev_principal (hf : ContinuousOn f s)
    (hs : IsCompact s) (hC : ∀ i ∈ s, f i ≠ 0) (c : F) : (fun _ => c) =O[𝓟 s] f :=
  (hf.isBigOWith_rev_principal hs hC c).isBigO


/-- The (scalar) product of a sequence that tends to zero with a bounded one also tends to zero. -/
lemma NormedField.tendsto_zero_smul_of_tendsto_zero_of_bounded {ι 𝕜 𝔸 : Type*}
    [NormedDivisionRing 𝕜] [NormedAddCommGroup 𝔸] [Module 𝕜 𝔸] [BoundedSMul 𝕜 𝔸] {l : Filter ι}
    {ε : ι → 𝕜} {f : ι → 𝔸} (hε : Tendsto ε l (𝓝 0)) (hf : IsBoundedUnder (· ≤ ·) l (norm ∘ f)) :
    Tendsto (ε • f) l (𝓝 0) := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    𝔸 : Type u_3
    inst✝³ : NormedDivisionRing 𝕜
    inst✝² : NormedAddCommGroup 𝔸
    inst✝¹ : Module 𝕜 𝔸
    inst✝ : BoundedSMul 𝕜 𝔸
    l : Filter ι
    ε : ι → 𝕜
    f : ι → 𝔸
    hε : Filter.Tendsto ε l (nhds 0)
    hf : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l (Function.comp Norm.no …
    ⊢ Filter.Tendsto (HSMul.hSMul ε f) l (nhds 0)
  -/
  rw [← isLittleO_one_iff 𝕜] at hε ⊢
  /-
    ι : Type u_1
    𝕜 : Type u_2
    𝔸 : Type u_3
    inst✝³ : NormedDivisionRing 𝕜
    inst✝² : NormedAddCommGroup 𝔸
    inst✝¹ : Module 𝕜 𝔸
    inst✝ : BoundedSMul 𝕜 𝔸
    l : Filter ι
    ε : ι → 𝕜
    f : ι → 𝔸
    hε : Asymptotics.IsLittleO l ε fun _x => 1
    hf : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) l (Function.comp Norm.no …
    ⊢ Asymptotics.IsLittleO l (HSMul.hSMul ε f) fun _x => 1
  -/
  simpa using IsLittleO.smul_isBigO hε (hf.isBigO_const (one_ne_zero : (1 : 𝕜) ≠ 0))
  /-
    🎉 no goals
  -/



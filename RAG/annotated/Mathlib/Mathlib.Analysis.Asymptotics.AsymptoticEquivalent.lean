/-- Two functions `u` and `v` are said to be asymptotically equivalent along a filter `l` when
    `u x - v x = o(v x)` as `x` converges along `l`. -/
def IsEquivalent (l : Filter α) (u v : α → β) :=
  (u - v) =o[l] v


@[inherit_doc] scoped notation:50 u " ~[" l:50 "] " v:50 => Asymptotics.IsEquivalent l u v


theorem IsEquivalent.isLittleO (h : u ~[l] v) : (u - v) =o[l] v := h


nonrec theorem IsEquivalent.isBigO (h : u ~[l] v) : u =O[l] v :=
  (IsBigO.congr_of_sub h.isBigO.symm).mp (isBigO_refl _ _)


theorem IsEquivalent.isBigO_symm (h : u ~[l] v) : v =O[l] u := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    u v : α → β
    l : Filter α
    h : Asymptotics.IsEquivalent l u v
    ⊢ Asymptotics.IsBigO l v u
  -/
  convert h.isLittleO.right_isBigO_add
  /-
    case h.e'_8.h
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    u v : α → β
    l : Filter α
    h : Asymptotics.IsEquivalent l u v
    x✝ : α
    ⊢ Eq (u x✝) (HAdd.hAdd (HSub.hSub u v x✝) (v x✝))
  -/
  simp
  /-
    🎉 no goals
  -/


theorem IsEquivalent.isTheta (h : u ~[l] v) : u =Θ[l] v :=
  ⟨h.isBigO, h.isBigO_symm⟩


theorem IsEquivalent.isTheta_symm (h : u ~[l] v) : v =Θ[l] u :=
  ⟨h.isBigO_symm, h.isBigO⟩


@[refl]
theorem IsEquivalent.refl : u ~[l] u := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    u : α → β
    l : Filter α
    ⊢ Asymptotics.IsEquivalent l u u
  -/
  rw [IsEquivalent, sub_self]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    u : α → β
    l : Filter α
    ⊢ Asymptotics.IsLittleO l 0 u
  -/
  exact isLittleO_zero _ _
  /-
    🎉 no goals
  -/


@[symm]
theorem IsEquivalent.symm (h : u ~[l] v) : v ~[l] u :=
  (h.isLittleO.trans_isBigO h.isBigO_symm).symm


@[trans]
theorem IsEquivalent.trans {l : Filter α} {u v w : α → β} (huv : u ~[l] v) (hvw : v ~[l] w) :
    u ~[l] w :=
  (huv.isLittleO.trans_isBigO hvw.isBigO).triangle hvw.isLittleO


theorem IsEquivalent.congr_left {u v w : α → β} {l : Filter α} (huv : u ~[l] v) (huw : u =ᶠ[l] w) :
    w ~[l] v :=
  huv.congr' (huw.sub (EventuallyEq.refl _ _)) (EventuallyEq.refl _ _)


theorem IsEquivalent.congr_right {u v w : α → β} {l : Filter α} (huv : u ~[l] v) (hvw : v =ᶠ[l] w) :
    u ~[l] w :=
  (huv.symm.congr_left hvw).symm


theorem isEquivalent_zero_iff_eventually_zero : u ~[l] 0 ↔ u =ᶠ[l] 0 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    u : α → β
    l : Filter α
    ⊢ Iff (Asymptotics.IsEquivalent l u 0) (l.EventuallyEq u 0)
  -/
  rw [IsEquivalent, sub_zero]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    u : α → β
    l : Filter α
    ⊢ Iff (Asymptotics.IsLittleO l u 0) (l.EventuallyEq u 0)
  -/
  exact isLittleO_zero_right_iff
  /-
    🎉 no goals
  -/


theorem isEquivalent_zero_iff_isBigO_zero : u ~[l] 0 ↔ u =O[l] (0 : α → β) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    u : α → β
    l : Filter α
    ⊢ Iff (Asymptotics.IsEquivalent l u 0) (Asymptotics.IsBigO l u 0)
  -/
  refine ⟨IsEquivalent.isBigO, fun h ↦ ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    u : α → β
    l : Filter α
    h : Asymptotics.IsBigO l u 0
    ⊢ Asymptotics.IsEquivalent l u 0
  -/
  rw [isEquivalent_zero_iff_eventually_zero, eventuallyEq_iff_exists_mem]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    u : α → β
    l : Filter α
    h : Asymptotics.IsBigO l u 0
    ⊢ Exists fun s => And (Membership.mem l s) (Set.EqOn u 0 s)
  -/
  exact ⟨{ x : α | u x = 0 }, isBigO_zero_right_iff.mp h, fun x hx ↦ hx⟩
  /-
    🎉 no goals
  -/


theorem isEquivalent_const_iff_tendsto {c : β} (h : c ≠ 0) :
    u ~[l] const _ c ↔ Tendsto u l (𝓝 c) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    u : α → β
    l : Filter α
    c : β
    h : Ne c 0
    ⊢ Iff (Asymptotics.IsEquivalent l u (Function.const α c)) (Filter.Tendsto u l  …
  -/
  simp (config := { unfoldPartialApp := true }) only [IsEquivalent, const, isLittleO_const_iff h]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    u : α → β
    l : Filter α
    c : β
    h : Ne c 0
    ⊢ Iff (Filter.Tendsto (HSub.hSub u fun x => c) l (nhds 0)) (Filter.Tendsto u l …
  -/
  constructor <;> intro h
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝ : NormedAddCommGroup β
      u : α → β
      l : Filter α
      c : β
      h✝ : Ne c 0
      h : Filter.Tendsto (HSub.hSub u fun x => c) l (nhds 0)
      ⊢ Filter.Tendsto u l (nhds c)
    -/
  · have := h.sub (tendsto_const_nhds (x := -c))
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝ : NormedAddCommGroup β
      u : α → β
      l : Filter α
      c : β
      h✝ : Ne c 0
      h : Filter.Tendsto (HSub.hSub u fun x => c) l (nhds 0)
      this : Filter.Tendsto (fun x => HSub.hSub (HSub.hSub u (fun x => c) x) (Neg.ne …
      ⊢ Filter.Tendsto u l (nhds c)
    -/
    simp only [Pi.sub_apply, sub_neg_eq_add, sub_add_cancel, zero_add] at this
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝ : NormedAddCommGroup β
      u : α → β
      l : Filter α
      c : β
      h✝ : Ne c 0
      h : Filter.Tendsto (HSub.hSub u fun x => c) l (nhds 0)
      this : Filter.Tendsto (fun x => u x) l (nhds c)
      ⊢ Filter.Tendsto u l (nhds c)
    -/
    exact this
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      inst✝ : NormedAddCommGroup β
      u : α → β
      l : Filter α
      c : β
      h✝ : Ne c 0
      h : Filter.Tendsto u l (nhds c)
      ⊢ Filter.Tendsto (HSub.hSub u fun x => c) l (nhds 0)
    -/
  · have := h.sub (tendsto_const_nhds (x := c))
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      inst✝ : NormedAddCommGroup β
      u : α → β
      l : Filter α
      c : β
      h✝ : Ne c 0
      h : Filter.Tendsto u l (nhds c)
      this : Filter.Tendsto (fun x => HSub.hSub (u x) c) l (nhds (HSub.hSub c c))
      ⊢ Filter.Tendsto (HSub.hSub u fun x => c) l (nhds 0)
    -/
    rwa [sub_self] at this
    /-
      🎉 no goals
    -/


theorem IsEquivalent.tendsto_const {c : β} (hu : u ~[l] const _ c) : Tendsto u l (𝓝 c) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    u : α → β
    l : Filter α
    c : β
    hu : Asymptotics.IsEquivalent l u (Function.const α c)
    ⊢ Filter.Tendsto u l (nhds c)
  -/
  rcases em <| c = 0 with rfl | h
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝ : NormedAddCommGroup β
      u : α → β
      l : Filter α
      hu : Asymptotics.IsEquivalent l u (Function.const α 0)
      ⊢ Filter.Tendsto u l (nhds 0)
    -/
  · exact (tendsto_congr' <| isEquivalent_zero_iff_eventually_zero.mp hu).mpr tendsto_const_nhds
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      inst✝ : NormedAddCommGroup β
      u : α → β
      l : Filter α
      c : β
      hu : Asymptotics.IsEquivalent l u (Function.const α c)
      h : Not (Eq c 0)
      ⊢ Filter.Tendsto u l (nhds c)
    -/
  · exact (isEquivalent_const_iff_tendsto h).mp hu
    /-
      🎉 no goals
    -/


theorem IsEquivalent.tendsto_nhds {c : β} (huv : u ~[l] v) (hu : Tendsto u l (𝓝 c)) :
    Tendsto v l (𝓝 c) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    u v : α → β
    l : Filter α
    c : β
    huv : Asymptotics.IsEquivalent l u v
    hu : Filter.Tendsto u l (nhds c)
    ⊢ Filter.Tendsto v l (nhds c)
  -/
  by_cases h : c = 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝ : NormedAddCommGroup β
      u v : α → β
      l : Filter α
      c : β
      huv : Asymptotics.IsEquivalent l u v
      hu : Filter.Tendsto u l (nhds c)
      h : Eq c 0
      ⊢ Filter.Tendsto v l (nhds c)
    -/
  · subst c
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝ : NormedAddCommGroup β
      u v : α → β
      l : Filter α
      huv : Asymptotics.IsEquivalent l u v
      hu : Filter.Tendsto u l (nhds 0)
      ⊢ Filter.Tendsto v l (nhds 0)
    -/
    rw [← isLittleO_one_iff ℝ] at hu ⊢
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝ : NormedAddCommGroup β
      u v : α → β
      l : Filter α
      huv : Asymptotics.IsEquivalent l u v
      hu : Asymptotics.IsLittleO l u fun _x => 1
      ⊢ Asymptotics.IsLittleO l v fun _x => 1
    -/
    simpa using (huv.symm.isLittleO.trans hu).add hu
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝ : NormedAddCommGroup β
      u v : α → β
      l : Filter α
      c : β
      huv : Asymptotics.IsEquivalent l u v
      hu : Filter.Tendsto u l (nhds c)
      h : Not (Eq c 0)
      ⊢ Filter.Tendsto v l (nhds c)
    -/
  · rw [← isEquivalent_const_iff_tendsto h] at hu ⊢
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝ : NormedAddCommGroup β
      u v : α → β
      l : Filter α
      c : β
      huv : Asymptotics.IsEquivalent l u v
      hu : Asymptotics.IsEquivalent l u (Function.const α c)
      h : Not (Eq c 0)
      ⊢ Asymptotics.IsEquivalent l v (Function.const α c)
    -/
    exact huv.symm.trans hu
    /-
      🎉 no goals
    -/


theorem IsEquivalent.tendsto_nhds_iff {c : β} (huv : u ~[l] v) :
    Tendsto u l (𝓝 c) ↔ Tendsto v l (𝓝 c) :=
  ⟨huv.tendsto_nhds, huv.symm.tendsto_nhds⟩


theorem IsEquivalent.add_isLittleO (huv : u ~[l] v) (hwv : w =o[l] v) : u + w ~[l] v := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    u v w : α → β
    l : Filter α
    huv : Asymptotics.IsEquivalent l u v
    hwv : Asymptotics.IsLittleO l w v
    ⊢ Asymptotics.IsEquivalent l (HAdd.hAdd u w) v
  -/
  simpa only [IsEquivalent, add_sub_right_comm] using huv.add hwv
  /-
    🎉 no goals
  -/


theorem IsEquivalent.sub_isLittleO (huv : u ~[l] v) (hwv : w =o[l] v) : u - w ~[l] v := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    u v w : α → β
    l : Filter α
    huv : Asymptotics.IsEquivalent l u v
    hwv : Asymptotics.IsLittleO l w v
    ⊢ Asymptotics.IsEquivalent l (HSub.hSub u w) v
  -/
  simpa only [sub_eq_add_neg] using huv.add_isLittleO hwv.neg_left
  /-
    🎉 no goals
  -/


theorem IsLittleO.add_isEquivalent (hu : u =o[l] w) (hv : v ~[l] w) : u + v ~[l] w :=
  add_comm v u ▸ hv.add_isLittleO hu


theorem IsLittleO.isEquivalent (huv : (u - v) =o[l] v) : u ~[l] v := huv


theorem IsEquivalent.neg (huv : u ~[l] v) : (fun x ↦ -u x) ~[l] fun x ↦ -v x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    u v : α → β
    l : Filter α
    huv : Asymptotics.IsEquivalent l u v
    ⊢ Asymptotics.IsEquivalent l (fun x => Neg.neg (u x)) fun x => Neg.neg (v x)
  -/
  rw [IsEquivalent]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    u v : α → β
    l : Filter α
    huv : Asymptotics.IsEquivalent l u v
    ⊢ Asymptotics.IsLittleO l (HSub.hSub (fun x => Neg.neg (u x)) fun x => Neg.neg …
  -/
  convert huv.isLittleO.neg_left.neg_right
  /-
    case h.e'_7.h
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    u v : α → β
    l : Filter α
    huv : Asymptotics.IsEquivalent l u v
    x✝ : α
    ⊢ Eq (HSub.hSub (fun x => Neg.neg (u x)) (fun x => Neg.neg (v x)) x✝) (Neg.neg …
  -/
  simp [neg_add_eq_sub]
  /-
    🎉 no goals
  -/


theorem isEquivalent_iff_exists_eq_mul :
    u ~[l] v ↔ ∃ (φ : α → β) (_ : Tendsto φ l (𝓝 1)), u =ᶠ[l] φ * v := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedField β
    u v : α → β
    l : Filter α
    ⊢ Iff (Asymptotics.IsEquivalent l u v) (Exists fun φ => Exists fun x => l.Even …
  -/
  rw [IsEquivalent, isLittleO_iff_exists_eq_mul]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedField β
    u v : α → β
    l : Filter α
    ⊢ Iff (Exists fun φ => And (Filter.Tendsto φ l (nhds 0)) (l.EventuallyEq (HSub …
  -/
  constructor <;> rintro ⟨φ, hφ, h⟩ <;> [refine ⟨φ + 1, ?_, ?_⟩; refine ⟨φ - 1, ?_, ?_⟩]
    /-
      case mp.intro.intro.refine_1
      α : Type u_1
      β : Type u_2
      inst✝ : NormedField β
      u v : α → β
      l : Filter α
      φ : α → β
      hφ : Filter.Tendsto φ l (nhds 0)
      h : l.EventuallyEq (HSub.hSub u v) (HMul.hMul φ v)
      ⊢ Filter.Tendsto (HAdd.hAdd φ 1) l (nhds 1)
    -/
  · conv in 𝓝 _ => rw [← zero_add (1 : β)]
    /-
      case mp.intro.intro.refine_1
      α : Type u_1
      β : Type u_2
      inst✝ : NormedField β
      u v : α → β
      l : Filter α
      φ : α → β
      hφ : Filter.Tendsto φ l (nhds 0)
      h : l.EventuallyEq (HSub.hSub u v) (HMul.hMul φ v)
      ⊢ Filter.Tendsto (HAdd.hAdd φ 1) l (nhds (HAdd.hAdd 0 1))
    -/
    exact hφ.add tendsto_const_nhds
    /-
      🎉 no goals
    -/
    /-
      case mp.intro.intro.refine_2
      α : Type u_1
      β : Type u_2
      inst✝ : NormedField β
      u v : α → β
      l : Filter α
      φ : α → β
      hφ : Filter.Tendsto φ l (nhds 0)
      h : l.EventuallyEq (HSub.hSub u v) (HMul.hMul φ v)
      ⊢ l.EventuallyEq u (HMul.hMul (HAdd.hAdd φ 1) v)
    -/
                                              /-
                                                🎉 no goals
                                              -/
  · convert h.add (EventuallyEq.refl l v) <;> simp [add_mul]
                                              /-
                                                🎉 no goals
                                              -/
    /-
      case mpr.intro.intro.refine_1
      α : Type u_1
      β : Type u_2
      inst✝ : NormedField β
      u v : α → β
      l : Filter α
      φ : α → β
      hφ : Filter.Tendsto φ l (nhds 1)
      h : l.EventuallyEq u (HMul.hMul φ v)
      ⊢ Filter.Tendsto (HSub.hSub φ 1) l (nhds 0)
    -/
  · conv in 𝓝 _ => rw [← sub_self (1 : β)]
    /-
      case mpr.intro.intro.refine_1
      α : Type u_1
      β : Type u_2
      inst✝ : NormedField β
      u v : α → β
      l : Filter α
      φ : α → β
      hφ : Filter.Tendsto φ l (nhds 1)
      h : l.EventuallyEq u (HMul.hMul φ v)
      ⊢ Filter.Tendsto (HSub.hSub φ 1) l (nhds (HSub.hSub 1 1))
    -/
    exact hφ.sub tendsto_const_nhds
    /-
      🎉 no goals
    -/
    /-
      case mpr.intro.intro.refine_2
      α : Type u_1
      β : Type u_2
      inst✝ : NormedField β
      u v : α → β
      l : Filter α
      φ : α → β
      hφ : Filter.Tendsto φ l (nhds 1)
      h : l.EventuallyEq u (HMul.hMul φ v)
      ⊢ l.EventuallyEq (HSub.hSub u v) (HMul.hMul (HSub.hSub φ 1) v)
    -/
  · convert h.sub (EventuallyEq.refl l v); simp [sub_mul]
                                           /-
                                             🎉 no goals
                                           -/


theorem IsEquivalent.exists_eq_mul (huv : u ~[l] v) :
    ∃ (φ : α → β) (_ : Tendsto φ l (𝓝 1)), u =ᶠ[l] φ * v :=
  isEquivalent_iff_exists_eq_mul.mp huv


theorem isEquivalent_of_tendsto_one (hz : ∀ᶠ x in l, v x = 0 → u x = 0)
    (huv : Tendsto (u / v) l (𝓝 1)) : u ~[l] v := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedField β
    u v : α → β
    l : Filter α
    hz : Filter.Eventually (fun x => Eq (v x) 0 → Eq (u x) 0) l
    huv : Filter.Tendsto (HDiv.hDiv u v) l (nhds 1)
    ⊢ Asymptotics.IsEquivalent l u v
  -/
  rw [isEquivalent_iff_exists_eq_mul]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedField β
    u v : α → β
    l : Filter α
    hz : Filter.Eventually (fun x => Eq (v x) 0 → Eq (u x) 0) l
    huv : Filter.Tendsto (HDiv.hDiv u v) l (nhds 1)
    ⊢ Exists fun φ => Exists fun x => l.EventuallyEq u (HMul.hMul φ v)
  -/
  exact ⟨u / v, huv, hz.mono fun x hz' ↦ (div_mul_cancel_of_imp hz').symm⟩
  /-
    🎉 no goals
  -/


theorem isEquivalent_of_tendsto_one' (hz : ∀ x, v x = 0 → u x = 0) (huv : Tendsto (u / v) l (𝓝 1)) :
    u ~[l] v :=
  isEquivalent_of_tendsto_one (Eventually.of_forall hz) huv


theorem isEquivalent_iff_tendsto_one (hz : ∀ᶠ x in l, v x ≠ 0) :
    u ~[l] v ↔ Tendsto (u / v) l (𝓝 1) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedField β
    u v : α → β
    l : Filter α
    hz : Filter.Eventually (fun x => Ne (v x) 0) l
    ⊢ Iff (Asymptotics.IsEquivalent l u v) (Filter.Tendsto (HDiv.hDiv u v) l (nhds …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝ : NormedField β
      u v : α → β
      l : Filter α
      hz : Filter.Eventually (fun x => Ne (v x) 0) l
      ⊢ Asymptotics.IsEquivalent l u v → Filter.Tendsto (HDiv.hDiv u v) l (nhds 1)
    -/
  · intro hequiv
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝ : NormedField β
      u v : α → β
      l : Filter α
      hz : Filter.Eventually (fun x => Ne (v x) 0) l
      hequiv : Asymptotics.IsEquivalent l u v
      ⊢ Filter.Tendsto (HDiv.hDiv u v) l (nhds 1)
    -/
    have := hequiv.isLittleO.tendsto_div_nhds_zero
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝ : NormedField β
      u v : α → β
      l : Filter α
      hz : Filter.Eventually (fun x => Ne (v x) 0) l
      hequiv : Asymptotics.IsEquivalent l u v
      this : Filter.Tendsto (fun x => HDiv.hDiv (HSub.hSub u v x) (v x)) l (nhds 0)
      ⊢ Filter.Tendsto (HDiv.hDiv u v) l (nhds 1)
    -/
    simp only [Pi.sub_apply, sub_div] at this
    have key : Tendsto (fun x ↦ v x / v x) l (𝓝 1) :=
      (tendsto_congr' <| hz.mono fun x hnz ↦ @div_self _ _ (v x) hnz).mpr tendsto_const_nhds
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝ : NormedField β
      u v : α → β
      l : Filter α
      hz : Filter.Eventually (fun x => Ne (v x) 0) l
      hequiv : Asymptotics.IsEquivalent l u v
      this : Filter.Tendsto (fun x => HSub.hSub (HDiv.hDiv (u x) (v x)) (HDiv.hDiv ( …
      key : Filter.Tendsto (fun x => HDiv.hDiv (v x) (v x)) l (nhds 1)
      ⊢ Filter.Tendsto (HDiv.hDiv u v) l (nhds 1)
    -/
    convert this.add key
      /-
        case h.e'_3.h
        α : Type u_1
        β : Type u_2
        inst✝ : NormedField β
        u v : α → β
        l : Filter α
        hz : Filter.Eventually (fun x => Ne (v x) 0) l
        hequiv : Asymptotics.IsEquivalent l u v
        this : Filter.Tendsto (fun x => HSub.hSub (HDiv.hDiv (u x) (v x)) (HDiv.hDiv ( …
        key : Filter.Tendsto (fun x => HDiv.hDiv (v x) (v x)) l (nhds 1)
        x✝ : α
        ⊢ Eq (HDiv.hDiv u v x✝) (HAdd.hAdd (HSub.hSub (HDiv.hDiv (u x✝) (v x✝)) (HDiv. …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case h.e'_5.h.e'_3
        α : Type u_1
        β : Type u_2
        inst✝ : NormedField β
        u v : α → β
        l : Filter α
        hz : Filter.Eventually (fun x => Ne (v x) 0) l
        hequiv : Asymptotics.IsEquivalent l u v
        this : Filter.Tendsto (fun x => HSub.hSub (HDiv.hDiv (u x) (v x)) (HDiv.hDiv ( …
        key : Filter.Tendsto (fun x => HDiv.hDiv (v x) (v x)) l (nhds 1)
        ⊢ Eq 1 (HAdd.hAdd 0 1)
      -/
    · norm_num
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      inst✝ : NormedField β
      u v : α → β
      l : Filter α
      hz : Filter.Eventually (fun x => Ne (v x) 0) l
      ⊢ Filter.Tendsto (HDiv.hDiv u v) l (nhds 1) → Asymptotics.IsEquivalent l u v
    -/
  · exact isEquivalent_of_tendsto_one (hz.mono fun x hnvz hz ↦ (hnvz hz).elim)
    /-
      🎉 no goals
    -/


theorem IsEquivalent.smul {α E 𝕜 : Type*} [NormedField 𝕜] [NormedAddCommGroup E] [NormedSpace 𝕜 E]
    {a b : α → 𝕜} {u v : α → E} {l : Filter α} (hab : a ~[l] b) (huv : u ~[l] v) :
    (fun x ↦ a x • u x) ~[l] fun x ↦ b x • v x := by
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    a b : α → 𝕜
    u v : α → E
    l : Filter α
    hab : Asymptotics.IsEquivalent l a b
    huv : Asymptotics.IsEquivalent l u v
    ⊢ Asymptotics.IsEquivalent l (fun x => HSMul.hSMul (a x) (u x)) fun x => HSMul …
  -/
  rcases hab.exists_eq_mul with ⟨φ, hφ, habφ⟩
  have : ((fun x ↦ a x • u x) - (fun x ↦ b x • v x)) =ᶠ[l] fun x ↦ b x • (φ x • u x - v x) := by
    -- Porting note: `convert` has become too strong, so we need to specify `using 1`.
    convert (habφ.comp₂ (· • ·) <| EventuallyEq.refl _ u).sub
      (EventuallyEq.refl _ fun x ↦ b x • v x) using 1
    ext
    rw [Pi.mul_apply, mul_comm, mul_smul, ← smul_sub]
  /-
    case intro.intro
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    a b : α → 𝕜
    u v : α → E
    l : Filter α
    hab : Asymptotics.IsEquivalent l a b
    huv : Asymptotics.IsEquivalent l u v
    φ : α → 𝕜
    hφ : Filter.Tendsto φ l (nhds 1)
    habφ : l.EventuallyEq a (HMul.hMul φ b)
    this : l.EventuallyEq (HSub.hSub (fun x => HSMul.hSMul (a x) (u x)) fun x => H …
    ⊢ Asymptotics.IsEquivalent l (fun x => HSMul.hSMul (a x) (u x)) fun x => HSMul …
  -/
  refine (isLittleO_congr this.symm <| EventuallyEq.rfl).mp ((isBigO_refl b l).smul_isLittleO ?_)
  /-
    case intro.intro
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    a b : α → 𝕜
    u v : α → E
    l : Filter α
    hab : Asymptotics.IsEquivalent l a b
    huv : Asymptotics.IsEquivalent l u v
    φ : α → 𝕜
    hφ : Filter.Tendsto φ l (nhds 1)
    habφ : l.EventuallyEq a (HMul.hMul φ b)
    this : l.EventuallyEq (HSub.hSub (fun x => HSMul.hSMul (a x) (u x)) fun x => H …
    ⊢ Asymptotics.IsLittleO l (fun x => HSub.hSub (HSMul.hSMul (φ x) (u x)) (v x)) v
  -/
  rcases huv.isBigO.exists_pos with ⟨C, hC, hCuv⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    a b : α → 𝕜
    u v : α → E
    l : Filter α
    hab : Asymptotics.IsEquivalent l a b
    huv : Asymptotics.IsEquivalent l u v
    φ : α → 𝕜
    hφ : Filter.Tendsto φ l (nhds 1)
    habφ : l.EventuallyEq a (HMul.hMul φ b)
    this : l.EventuallyEq (HSub.hSub (fun x => HSMul.hSMul (a x) (u x)) fun x => H …
    C : Real
    hC : GT.gt C 0
    hCuv : Asymptotics.IsBigOWith C l u v
    ⊢ Asymptotics.IsLittleO l (fun x => HSub.hSub (HSMul.hSMul (φ x) (u x)) (v x)) v
  -/
  rw [IsEquivalent] at *
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    a b : α → 𝕜
    u v : α → E
    l : Filter α
    hab : Asymptotics.IsLittleO l (HSub.hSub a b) b
    huv : Asymptotics.IsLittleO l (HSub.hSub u v) v
    φ : α → 𝕜
    hφ : Filter.Tendsto φ l (nhds 1)
    habφ : l.EventuallyEq a (HMul.hMul φ b)
    this : l.EventuallyEq (HSub.hSub (fun x => HSMul.hSMul (a x) (u x)) fun x => H …
    C : Real
    hC : GT.gt C 0
    hCuv : Asymptotics.IsBigOWith C l u v
    ⊢ Asymptotics.IsLittleO l (fun x => HSub.hSub (HSMul.hSMul (φ x) (u x)) (v x)) v
  -/
  rw [isLittleO_iff] at *
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    a b : α → 𝕜
    u v : α → E
    l : Filter α
    hab : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    huv : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    φ : α → 𝕜
    hφ : Filter.Tendsto φ l (nhds 1)
    habφ : l.EventuallyEq a (HMul.hMul φ b)
    this : l.EventuallyEq (HSub.hSub (fun x => HSMul.hSMul (a x) (u x)) fun x => H …
    C : Real
    hC : GT.gt C 0
    hCuv : Asymptotics.IsBigOWith C l u v
    ⊢ ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm (HSub …
  -/
  rw [IsBigOWith] at hCuv
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    a b : α → 𝕜
    u v : α → E
    l : Filter α
    hab : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    huv : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    φ : α → 𝕜
    hφ : Filter.Tendsto φ l (nhds 1)
    habφ : l.EventuallyEq a (HMul.hMul φ b)
    this : l.EventuallyEq (HSub.hSub (fun x => HSMul.hSMul (a x) (u x)) fun x => H …
    C : Real
    hC : GT.gt C 0
    hCuv : Filter.Eventually (fun x => LE.le (Norm.norm (u x)) (HMul.hMul C (Norm. …
    ⊢ ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm (HSub …
  -/
  simp only [Metric.tendsto_nhds, dist_eq_norm] at hφ
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    a b : α → 𝕜
    u v : α → E
    l : Filter α
    hab : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    huv : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    φ : α → 𝕜
    habφ : l.EventuallyEq a (HMul.hMul φ b)
    this : l.EventuallyEq (HSub.hSub (fun x => HSMul.hSMul (a x) (u x)) fun x => H …
    C : Real
    hC : GT.gt C 0
    hCuv : Filter.Eventually (fun x => LE.le (Norm.norm (u x)) (HMul.hMul C (Norm. …
    hφ : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Norm.norm (H …
    ⊢ ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm (HSub …
  -/
  intro c hc
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    a b : α → 𝕜
    u v : α → E
    l : Filter α
    hab : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    huv : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    φ : α → 𝕜
    habφ : l.EventuallyEq a (HMul.hMul φ b)
    this : l.EventuallyEq (HSub.hSub (fun x => HSMul.hSMul (a x) (u x)) fun x => H …
    C : Real
    hC : GT.gt C 0
    hCuv : Filter.Eventually (fun x => LE.le (Norm.norm (u x)) (HMul.hMul C (Norm. …
    hφ : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Norm.norm (H …
    c : Real
    hc : LT.lt 0 c
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (HSMul.hSMul (φ x) ( …
  -/
  specialize hφ (c / 2 / C) (div_pos (div_pos hc zero_lt_two) hC)
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    a b : α → 𝕜
    u v : α → E
    l : Filter α
    hab : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    huv : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    φ : α → 𝕜
    habφ : l.EventuallyEq a (HMul.hMul φ b)
    this : l.EventuallyEq (HSub.hSub (fun x => HSMul.hSMul (a x) (u x)) fun x => H …
    C : Real
    hC : GT.gt C 0
    hCuv : Filter.Eventually (fun x => LE.le (Norm.norm (u x)) (HMul.hMul C (Norm. …
    c : Real
    hc : LT.lt 0 c
    hφ : Filter.Eventually (fun x => LT.lt (Norm.norm (HSub.hSub (φ x) 1)) (HDiv.h …
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (HSMul.hSMul (φ x) ( …
  -/
  specialize huv (div_pos hc zero_lt_two)
  /-
    case intro.intro.intro.intro
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_3
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    a b : α → 𝕜
    u v : α → E
    l : Filter α
    hab : ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x => LE.le (Norm.norm ( …
    φ : α → 𝕜
    habφ : l.EventuallyEq a (HMul.hMul φ b)
    this : l.EventuallyEq (HSub.hSub (fun x => HSMul.hSMul (a x) (u x)) fun x => H …
    C : Real
    hC : GT.gt C 0
    hCuv : Filter.Eventually (fun x => LE.le (Norm.norm (u x)) (HMul.hMul C (Norm. …
    c : Real
    hc : LT.lt 0 c
    hφ : Filter.Eventually (fun x => LT.lt (Norm.norm (HSub.hSub (φ x) 1)) (HDiv.h …
    huv : Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub u v x)) (HMul.hM …
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub (HSMul.hSMul (φ x) ( …
  -/
  refine hφ.mp (huv.mp <| hCuv.mono fun x hCuvx huvx hφx ↦ ?_)
  have key :=
    calc
      ‖φ x - 1‖ * ‖u x‖ ≤ c / 2 / C * ‖u x‖ := by gcongr
      _ ≤ c / 2 / C * (C * ‖v x‖) := by gcongr
      _ = c / 2 * ‖v x‖ := by
        field_simp [hC.ne.symm]
        ring
  calc
    ‖((fun x : α ↦ φ x • u x) - v) x‖ = ‖(φ x - 1) • u x + (u x - v x)‖ := by
      simp [sub_smul, sub_add]
    _ ≤ ‖(φ x - 1) • u x‖ + ‖u x - v x‖ := norm_add_le _ _
    _ = ‖φ x - 1‖ * ‖u x‖ + ‖u x - v x‖ := by rw [norm_smul]
    _ ≤ c / 2 * ‖v x‖ + ‖u x - v x‖ := by gcongr
    _ ≤ c / 2 * ‖v x‖ + c / 2 * ‖v x‖ := by gcongr; exact huvx
    _ = c * ‖v x‖ := by ring


theorem IsEquivalent.mul (htu : t ~[l] u) (hvw : v ~[l] w) : t * v ~[l] u * w :=
  htu.smul hvw


theorem IsEquivalent.inv (huv : u ~[l] v) : (fun x ↦ (u x)⁻¹) ~[l] fun x ↦ (v x)⁻¹ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedField β
    u v : α → β
    l : Filter α
    huv : Asymptotics.IsEquivalent l u v
    ⊢ Asymptotics.IsEquivalent l (fun x => Inv.inv (u x)) fun x => Inv.inv (v x)
  -/
  rw [isEquivalent_iff_exists_eq_mul] at *
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedField β
    u v : α → β
    l : Filter α
    huv : Exists fun φ => Exists fun x => l.EventuallyEq u (HMul.hMul φ v)
    ⊢ Exists fun φ => Exists fun x => l.EventuallyEq (fun x => Inv.inv (u x)) (HMu …
  -/
  rcases huv with ⟨φ, hφ, h⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝ : NormedField β
    u v : α → β
    l : Filter α
    φ : α → β
    hφ : Filter.Tendsto φ l (nhds 1)
    h : l.EventuallyEq u (HMul.hMul φ v)
    ⊢ Exists fun φ => Exists fun x => l.EventuallyEq (fun x => Inv.inv (u x)) (HMu …
  -/
  rw [← inv_one]
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝ : NormedField β
    u v : α → β
    l : Filter α
    φ : α → β
    hφ : Filter.Tendsto φ l (nhds 1)
    h : l.EventuallyEq u (HMul.hMul φ v)
    ⊢ Exists fun φ => Exists fun x => l.EventuallyEq (fun x => Inv.inv (u x)) (HMu …
  -/
  refine ⟨fun x ↦ (φ x)⁻¹, Tendsto.inv₀ hφ (by norm_num), ?_⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝ : NormedField β
    u v : α → β
    l : Filter α
    φ : α → β
    hφ : Filter.Tendsto φ l (nhds 1)
    h : l.EventuallyEq u (HMul.hMul φ v)
    ⊢ l.EventuallyEq (fun x => Inv.inv (u x)) (HMul.hMul (fun x => Inv.inv (φ x))  …
  -/
  convert h.inv
  /-
    case h.e'_5.h
    α : Type u_1
    β : Type u_2
    inst✝ : NormedField β
    u v : α → β
    l : Filter α
    φ : α → β
    hφ : Filter.Tendsto φ l (nhds 1)
    h : l.EventuallyEq u (HMul.hMul φ v)
    x✝ : α
    ⊢ Eq (HMul.hMul (fun x => Inv.inv (φ x)) (fun x => Inv.inv (v x)) x✝) (Inv.inv …
  -/
  simp [mul_comm]
  /-
    🎉 no goals
  -/


theorem IsEquivalent.div (htu : t ~[l] u) (hvw : v ~[l] w) :
    (fun x ↦ t x / v x) ~[l] fun x ↦ u x / w x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedField β
    t u v w : α → β
    l : Filter α
    htu : Asymptotics.IsEquivalent l t u
    hvw : Asymptotics.IsEquivalent l v w
    ⊢ Asymptotics.IsEquivalent l (fun x => HDiv.hDiv (t x) (v x)) fun x => HDiv.hD …
  -/
  simpa only [div_eq_mul_inv] using htu.mul hvw.inv
  /-
    🎉 no goals
  -/


theorem IsEquivalent.tendsto_atTop [OrderTopology β] (huv : u ~[l] v) (hu : Tendsto u l atTop) :
    Tendsto v l atTop :=
  let ⟨φ, hφ, h⟩ := huv.symm.exists_eq_mul
  Tendsto.congr' h.symm (mul_comm u φ ▸ hu.atTop_mul zero_lt_one hφ)


theorem IsEquivalent.tendsto_atTop_iff [OrderTopology β] (huv : u ~[l] v) :
    Tendsto u l atTop ↔ Tendsto v l atTop :=
  ⟨huv.tendsto_atTop, huv.symm.tendsto_atTop⟩


theorem IsEquivalent.tendsto_atBot [OrderTopology β] (huv : u ~[l] v) (hu : Tendsto u l atBot) :
    Tendsto v l atBot := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : NormedLinearOrderedField β
    u v : α → β
    l : Filter α
    inst✝ : OrderTopology β
    huv : Asymptotics.IsEquivalent l u v
    hu : Filter.Tendsto u l Filter.atBot
    ⊢ Filter.Tendsto v l Filter.atBot
  -/
  convert tendsto_neg_atTop_atBot.comp (huv.neg.tendsto_atTop <| tendsto_neg_atBot_atTop.comp hu)
  /-
    case h.e'_3
    α : Type u_1
    β : Type u_2
    inst✝¹ : NormedLinearOrderedField β
    u v : α → β
    l : Filter α
    inst✝ : OrderTopology β
    huv : Asymptotics.IsEquivalent l u v
    hu : Filter.Tendsto u l Filter.atBot
    ⊢ Eq v (Function.comp Neg.neg fun x => Neg.neg (v x))
  -/
  ext
  /-
    case h.e'_3.h
    α : Type u_1
    β : Type u_2
    inst✝¹ : NormedLinearOrderedField β
    u v : α → β
    l : Filter α
    inst✝ : OrderTopology β
    huv : Asymptotics.IsEquivalent l u v
    hu : Filter.Tendsto u l Filter.atBot
    x✝ : α
    ⊢ Eq (v x✝) (Function.comp Neg.neg (fun x => Neg.neg (v x)) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem IsEquivalent.tendsto_atBot_iff [OrderTopology β] (huv : u ~[l] v) :
    Tendsto u l atBot ↔ Tendsto v l atBot :=
  ⟨huv.tendsto_atBot, huv.symm.tendsto_atBot⟩


theorem Filter.EventuallyEq.isEquivalent {u v : α → β} (h : u =ᶠ[l] v) : u ~[l] v :=
  IsEquivalent.congr_right (isLittleO_refl_left _ _) h


@[trans]
theorem Filter.EventuallyEq.trans_isEquivalent {f g₁ g₂ : α → β} (h : f =ᶠ[l] g₁)
    (h₂ : g₁ ~[l] g₂) : f ~[l] g₂ :=
  h.isEquivalent.trans h₂


instance transIsEquivalentIsEquivalent :
    @Trans (α → β) (α → β) (α → β) (IsEquivalent l) (IsEquivalent l) (IsEquivalent l) where
  trans := IsEquivalent.trans


instance transEventuallyEqIsEquivalent :
    @Trans (α → β) (α → β) (α → β) (EventuallyEq l) (IsEquivalent l) (IsEquivalent l) where
  trans := EventuallyEq.trans_isEquivalent


@[trans]
theorem IsEquivalent.trans_eventuallyEq {f g₁ g₂ : α → β} (h : f ~[l] g₁)
    (h₂ : g₁ =ᶠ[l] g₂) : f ~[l] g₂ :=
  h.trans h₂.isEquivalent


instance transIsEquivalentEventuallyEq :
    @Trans (α → β) (α → β) (α → β) (IsEquivalent l) (EventuallyEq l) (IsEquivalent l) where
  trans := IsEquivalent.trans_eventuallyEq


@[trans]
theorem IsEquivalent.trans_isBigO {f g₁ : α → β} {g₂ : α → β₂} (h : f ~[l] g₁) (h₂ : g₁ =O[l] g₂) :
    f =O[l] g₂ :=
  IsBigO.trans h.isBigO h₂


instance transIsEquivalentIsBigO :
    @Trans (α → β) (α → β) (α → β₂) (IsEquivalent l) (IsBigO l) (IsBigO l) where
  trans := IsEquivalent.trans_isBigO


@[trans]
theorem IsBigO.trans_isEquivalent {f : α → β₂} {g₁ g₂ : α → β} (h : f =O[l] g₁) (h₂ : g₁ ~[l] g₂) :
    f =O[l] g₂ :=
  IsBigO.trans h h₂.isBigO


instance transIsBigOIsEquivalent :
    @Trans (α → β₂) (α → β) (α → β) (IsBigO l) (IsEquivalent l) (IsBigO l) where
  trans := IsBigO.trans_isEquivalent


@[trans]
theorem IsEquivalent.trans_isLittleO {f g₁ : α → β} {g₂ : α → β₂} (h : f ~[l] g₁)
    (h₂ : g₁ =o[l] g₂) : f =o[l] g₂ :=
  IsBigO.trans_isLittleO h.isBigO h₂


instance transIsEquivalentIsLittleO :
    @Trans (α → β) (α → β) (α → β₂) (IsEquivalent l) (IsLittleO l) (IsLittleO l) where
  trans := IsEquivalent.trans_isLittleO


@[trans]
theorem IsLittleO.trans_isEquivalent {f : α → β₂} {g₁ g₂ : α → β} (h : f =o[l] g₁)
    (h₂ : g₁ ~[l] g₂) : f =o[l] g₂ :=
  IsLittleO.trans_isBigO h h₂.isBigO


instance transIsLittleOIsEquivalent :
    @Trans (α → β₂) (α → β) (α → β) (IsLittleO l) (IsEquivalent l) (IsLittleO l) where
  trans := IsLittleO.trans_isEquivalent


@[trans]
theorem IsEquivalent.trans_isTheta {f g₁ : α → β} {g₂ : α → β₂} (h : f ~[l] g₁)
    (h₂ : g₁ =Θ[l] g₂) : f =Θ[l] g₂ :=
  IsTheta.trans h.isTheta h₂


instance transIsEquivalentIsTheta :
    @Trans (α → β) (α → β) (α → β₂) (IsEquivalent l) (IsTheta l) (IsTheta l) where
  trans := IsEquivalent.trans_isTheta


@[trans]
theorem IsTheta.trans_isEquivalent {f : α → β₂} {g₁ g₂ : α → β} (h : f =Θ[l] g₁)
    (h₂ : g₁ ~[l] g₂) : f =Θ[l] g₂ :=
  IsTheta.trans h h₂.isTheta


instance transIsThetaIsEquivalent :
    @Trans (α → β₂) (α → β) (α → β) (IsTheta l) (IsEquivalent l) (IsTheta l) where
  trans := IsTheta.trans_isEquivalent



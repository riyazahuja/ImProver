/-- The (extended real valued) variation of a function `f` on a set `s` inside a linear order is
the supremum of the sum of `edist (f (u (i+1))) (f (u i))` over all finite increasing
sequences `u` in `s`. -/
noncomputable def eVariationOn (f : α → E) (s : Set α) : ℝ≥0∞ :=
  ⨆ p : ℕ × { u : ℕ → α // Monotone u ∧ ∀ i, u i ∈ s },
    ∑ i ∈ Finset.range p.1, edist (f (p.2.1 (i + 1))) (f (p.2.1 i))


/-- A function has bounded variation on a set `s` if its total variation there is finite. -/
def BoundedVariationOn (f : α → E) (s : Set α) :=
  eVariationOn f s ≠ ∞


/-- A function has locally bounded variation on a set `s` if, given any interval `[a, b]` with
endpoints in `s`, then the function has finite variation on `s ∩ [a, b]`. -/
def LocallyBoundedVariationOn (f : α → E) (s : Set α) :=
  ∀ a b, a ∈ s → b ∈ s → BoundedVariationOn f (s ∩ Icc a b)


theorem nonempty_monotone_mem {s : Set α} (hs : s.Nonempty) :
    Nonempty { u // Monotone u ∧ ∀ i : ℕ, u i ∈ s } := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    hs : s.Nonempty
    ⊢ Nonempty (Subtype fun u => And (Monotone u) (∀ (i : Nat), Membership.mem s ( …
  -/
  obtain ⟨x, hx⟩ := hs
  /-
    case intro
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    x : α
    hx : Membership.mem s x
    ⊢ Nonempty (Subtype fun u => And (Monotone u) (∀ (i : Nat), Membership.mem s ( …
  -/
  exact ⟨⟨fun _ => x, fun i j _ => le_rfl, fun _ => hx⟩⟩
  /-
    🎉 no goals
  -/


theorem eq_of_edist_zero_on {f f' : α → E} {s : Set α} (h : ∀ ⦃x⦄, x ∈ s → edist (f x) (f' x) = 0) :
    eVariationOn f s = eVariationOn f' s := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f f' : α → E
    s : Set α
    h : ∀ ⦃x : α⦄, Membership.mem s x → Eq (EDist.edist (f x) (f' x)) 0
    ⊢ Eq (eVariationOn f s) (eVariationOn f' s)
  -/
  dsimp only [eVariationOn]
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f f' : α → E
    s : Set α
    h : ∀ ⦃x : α⦄, Membership.mem s x → Eq (EDist.edist (f x) (f' x)) 0
    ⊢ Eq (iSup fun p => (Finset.range p.1).sum fun i => EDist.edist (f (↑p.2 (HAdd …
  -/
  congr 1 with p : 1
  /-
    case e_s.h
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f f' : α → E
    s : Set α
    h : ∀ ⦃x : α⦄, Membership.mem s x → Eq (EDist.edist (f x) (f' x)) 0
    p : Prod Nat (Subtype fun u => And (Monotone u) (∀ (i : Nat), Membership.mem s …
    ⊢ Eq ((Finset.range p.1).sum fun i => EDist.edist (f (↑p.2 (HAdd.hAdd i 1))) ( …
  -/
  congr 1 with i : 1
  /-
    case e_s.h.e_f.h
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f f' : α → E
    s : Set α
    h : ∀ ⦃x : α⦄, Membership.mem s x → Eq (EDist.edist (f x) (f' x)) 0
    p : Prod Nat (Subtype fun u => And (Monotone u) (∀ (i : Nat), Membership.mem s …
    i : Nat
    ⊢ Eq (EDist.edist (f (↑p.2 (HAdd.hAdd i 1))) (f (↑p.2 i))) (EDist.edist (f' (↑ …
  -/
  rw [edist_congr_right (h <| p.snd.prop.2 (i + 1)), edist_congr_left (h <| p.snd.prop.2 i)]
  /-
    🎉 no goals
  -/


theorem eq_of_eqOn {f f' : α → E} {s : Set α} (h : EqOn f f' s) :
    eVariationOn f s = eVariationOn f' s :=
                                     /-
                                       α : Type u_1
                                       inst✝¹ : LinearOrder α
                                       E : Type u_2
                                       inst✝ : PseudoEMetricSpace E
                                       f f' : α → E
                                       s : Set α
                                       h : Set.EqOn f f' s
                                       x : α
                                       xs : Membership.mem s x
                                       ⊢ Eq (EDist.edist (f x) (f' x)) 0
                                     -/
  eq_of_edist_zero_on fun x xs => by rw [h xs, edist_self]
                                     /-
                                       🎉 no goals
                                     -/


theorem sum_le (f : α → E) {s : Set α} (n : ℕ) {u : ℕ → α} (hu : Monotone u) (us : ∀ i, u i ∈ s) :
    (∑ i ∈ Finset.range n, edist (f (u (i + 1))) (f (u i))) ≤ eVariationOn f s :=
  le_iSup_of_le ⟨n, u, hu, us⟩ le_rfl


theorem sum_le_of_monotoneOn_Icc (f : α → E) {s : Set α} {m n : ℕ} {u : ℕ → α}
    (hu : MonotoneOn u (Icc m n)) (us : ∀ i ∈ Icc m n, u i ∈ s) :
    (∑ i ∈ Finset.Ico m n, edist (f (u (i + 1))) (f (u i))) ≤ eVariationOn f s := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    m n : Nat
    u : Nat → α
    hu : MonotoneOn u (Set.Icc m n)
    us : ∀ (i : Nat), Membership.mem (Set.Icc m n) i → Membership.mem s (u i)
    ⊢ LE.le ((Finset.Ico m n).sum fun i => EDist.edist (f (u (HAdd.hAdd i 1))) (f  …
  -/
  rcases le_total n m with hnm | hmn
    /-
      case inl
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      m n : Nat
      u : Nat → α
      hu : MonotoneOn u (Set.Icc m n)
      us : ∀ (i : Nat), Membership.mem (Set.Icc m n) i → Membership.mem s (u i)
      hnm : LE.le n m
      ⊢ LE.le ((Finset.Ico m n).sum fun i => EDist.edist (f (u (HAdd.hAdd i 1))) (f  …
    -/
  · simp [Finset.Ico_eq_empty_of_le hnm]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    m n : Nat
    u : Nat → α
    hu : MonotoneOn u (Set.Icc m n)
    us : ∀ (i : Nat), Membership.mem (Set.Icc m n) i → Membership.mem s (u i)
    hmn : LE.le m n
    ⊢ LE.le ((Finset.Ico m n).sum fun i => EDist.edist (f (u (HAdd.hAdd i 1))) (f  …
  -/
  let π := projIcc m n hmn
  /-
    case inr
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    m n : Nat
    u : Nat → α
    hu : MonotoneOn u (Set.Icc m n)
    us : ∀ (i : Nat), Membership.mem (Set.Icc m n) i → Membership.mem s (u i)
    hmn : LE.le m n
    π : Nat → ↑(Set.Icc m n) := Set.projIcc m n hmn
    ⊢ LE.le ((Finset.Ico m n).sum fun i => EDist.edist (f (u (HAdd.hAdd i 1))) (f  …
  -/
  let v i := u (π i)
  calc
    ∑ i ∈ Finset.Ico m n, edist (f (u (i + 1))) (f (u i))
        = ∑ i ∈ Finset.Ico m n, edist (f (v (i + 1))) (f (v i)) :=
      Finset.sum_congr rfl fun i hi ↦ by
        rw [Finset.mem_Ico] at hi
        simp only [v, π, projIcc_of_mem hmn ⟨hi.1, hi.2.le⟩,
          projIcc_of_mem hmn ⟨hi.1.trans i.le_succ, hi.2⟩]
    _ ≤ ∑ i ∈ Finset.range n, edist (f (v (i + 1))) (f (v i)) :=
      Finset.sum_mono_set _ (Nat.Iio_eq_range ▸ Finset.Ico_subset_Iio_self)
    _ ≤ eVariationOn f s :=
      sum_le _ _ (fun i j h ↦ hu (π i).2 (π j).2 (monotone_projIcc hmn h)) fun i ↦ us _ (π i).2


theorem sum_le_of_monotoneOn_Iic (f : α → E) {s : Set α} {n : ℕ} {u : ℕ → α}
    (hu : MonotoneOn u (Iic n)) (us : ∀ i ≤ n, u i ∈ s) :
    (∑ i ∈ Finset.range n, edist (f (u (i + 1))) (f (u i))) ≤ eVariationOn f s := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    n : Nat
    u : Nat → α
    hu : MonotoneOn u (Set.Iic n)
    us : ∀ (i : Nat), LE.le i n → Membership.mem s (u i)
    ⊢ LE.le ((Finset.range n).sum fun i => EDist.edist (f (u (HAdd.hAdd i 1))) (f  …
  -/
  simpa using sum_le_of_monotoneOn_Icc f (m := 0) (hu.mono Icc_subset_Iic_self) fun i hi ↦ us i hi.2
  /-
    🎉 no goals
  -/


theorem mono (f : α → E) {s t : Set α} (hst : t ⊆ s) : eVariationOn f t ≤ eVariationOn f s := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s t : Set α
    hst : HasSubset.Subset t s
    ⊢ LE.le (eVariationOn f t) (eVariationOn f s)
  -/
  apply iSup_le _
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s t : Set α
    hst : HasSubset.Subset t s
    ⊢ ∀ (i : Prod Nat (Subtype fun u => And (Monotone u) (∀ (i : Nat), Membership. …
  -/
  rintro ⟨n, ⟨u, hu, ut⟩⟩
  /-
    case mk.mk.intro
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s t : Set α
    hst : HasSubset.Subset t s
    n : Nat
    u : Nat → α
    hu : Monotone u
    ut : ∀ (i : Nat), Membership.mem t (u i)
    ⊢ LE.le ((Finset.range { fst := n, snd := ⟨u, ⋯⟩ }.1).sum fun i => EDist.edist …
  -/
  exact sum_le f n hu fun i => hst (ut i)
  /-
    🎉 no goals
  -/


theorem _root_.BoundedVariationOn.mono {f : α → E} {s : Set α} (h : BoundedVariationOn f s)
    {t : Set α} (ht : t ⊆ s) : BoundedVariationOn f t :=
  ne_top_of_le_ne_top h (eVariationOn.mono f ht)


theorem _root_.BoundedVariationOn.locallyBoundedVariationOn {f : α → E} {s : Set α}
    (h : BoundedVariationOn f s) : LocallyBoundedVariationOn f s := fun _ _ _ _ =>
  h.mono inter_subset_left


theorem edist_le (f : α → E) {s : Set α} {x y : α} (hx : x ∈ s) (hy : y ∈ s) :
    edist (f x) (f y) ≤ eVariationOn f s := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    x y : α
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ LE.le (EDist.edist (f x) (f y)) (eVariationOn f s)
  -/
  wlog hxy : y ≤ x generalizing x y
    /-
      case inr
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      x y : α
      hx : Membership.mem s x
      hy : Membership.mem s y
      this : ∀ {x y : α}, Membership.mem s x → Membership.mem s y → LE.le y x → LE.l …
      hxy : Not (LE.le y x)
      ⊢ LE.le (EDist.edist (f x) (f y)) (eVariationOn f s)
    -/
  · rw [edist_comm]
    /-
      case inr
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      x y : α
      hx : Membership.mem s x
      hy : Membership.mem s y
      this : ∀ {x y : α}, Membership.mem s x → Membership.mem s y → LE.le y x → LE.l …
      hxy : Not (LE.le y x)
      ⊢ LE.le (EDist.edist (f y) (f x)) (eVariationOn f s)
    -/
    exact this hy hx (le_of_not_le hxy)
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    x y : α
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxy : LE.le y x
    ⊢ LE.le (EDist.edist (f x) (f y)) (eVariationOn f s)
  -/
  let u : ℕ → α := fun n => if n = 0 then y else x
  have hu : Monotone u := monotone_nat_of_le_succ fun
  | 0 => hxy
  | (_ + 1) => le_rfl
  have us : ∀ i, u i ∈ s := fun
  | 0 => hy
  | (_ + 1) => hx
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    x y : α
    hx : Membership.mem s x
    hy : Membership.mem s y
    hxy : LE.le y x
    u : Nat → α := fun n => ite (Eq n 0) y x
    hu : Monotone u
    us : ∀ (i : Nat), Membership.mem s (u i)
    ⊢ LE.le (EDist.edist (f x) (f y)) (eVariationOn f s)
  -/
  simpa only [Finset.sum_range_one] using sum_le f 1 hu us
  /-
    🎉 no goals
  -/


theorem eq_zero_iff (f : α → E) {s : Set α} :
    eVariationOn f s = 0 ↔ ∀ x ∈ s, ∀ y ∈ s, edist (f x) (f y) = 0 := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    ⊢ Iff (Eq (eVariationOn f s) 0) (∀ (x : α), Membership.mem s x → ∀ (y : α), Me …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      ⊢ Eq (eVariationOn f s) 0 → ∀ (x : α), Membership.mem s x → ∀ (y : α), Members …
    -/
  · rintro h x xs y ys
    /-
      case mp
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      h : Eq (eVariationOn f s) 0
      x : α
      xs : Membership.mem s x
      y : α
      ys : Membership.mem s y
      ⊢ Eq (EDist.edist (f x) (f y)) 0
    -/
    rw [← le_zero_iff, ← h]
    /-
      case mp
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      h : Eq (eVariationOn f s) 0
      x : α
      xs : Membership.mem s x
      y : α
      ys : Membership.mem s y
      ⊢ LE.le (EDist.edist (f x) (f y)) (eVariationOn f s)
    -/
    exact edist_le f xs ys
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      ⊢ (∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → Eq (EDist.e …
    -/
  · rintro h
    /-
      case mpr
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → Eq (EDist. …
      ⊢ Eq (eVariationOn f s) 0
    -/
    dsimp only [eVariationOn]
    /-
      case mpr
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → Eq (EDist. …
      ⊢ Eq (iSup fun p => (Finset.range p.1).sum fun i => EDist.edist (f (↑p.2 (HAdd …
    -/
    rw [ENNReal.iSup_eq_zero]
    /-
      case mpr
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → Eq (EDist. …
      ⊢ ∀ (i : Prod Nat (Subtype fun u => And (Monotone u) (∀ (i : Nat), Membership. …
    -/
    rintro ⟨n, u, um, us⟩
    /-
      case mpr.mk.mk.intro
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → Eq (EDist. …
      n : Nat
      u : Nat → α
      um : Monotone u
      us : ∀ (i : Nat), Membership.mem s (u i)
      ⊢ Eq ((Finset.range { fst := n, snd := ⟨u, ⋯⟩ }.1).sum fun i => EDist.edist (f …
    -/
    exact Finset.sum_eq_zero fun i _ => h _ (us i.succ) _ (us i)
    /-
      🎉 no goals
    -/


theorem constant_on {f : α → E} {s : Set α} (hf : (f '' s).Subsingleton) :
    eVariationOn f s = 0 := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : (Set.image f s).Subsingleton
    ⊢ Eq (eVariationOn f s) 0
  -/
  rw [eq_zero_iff]
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : (Set.image f s).Subsingleton
    ⊢ ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → Eq (EDist.ed …
  -/
  rintro x xs y ys
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : (Set.image f s).Subsingleton
    x : α
    xs : Membership.mem s x
    y : α
    ys : Membership.mem s y
    ⊢ Eq (EDist.edist (f x) (f y)) 0
  -/
  rw [hf ⟨x, xs, rfl⟩ ⟨y, ys, rfl⟩, edist_self]
  /-
    🎉 no goals
  -/


@[simp]
protected theorem subsingleton (f : α → E) {s : Set α} (hs : s.Subsingleton) :
    eVariationOn f s = 0 :=
  constant_on (hs.image f)


theorem lowerSemicontinuous_aux {ι : Type*} {F : ι → α → E} {p : Filter ι} {f : α → E} {s : Set α}
    (Ffs : ∀ x ∈ s, Tendsto (fun i => F i x) p (𝓝 (f x))) {v : ℝ≥0∞} (hv : v < eVariationOn f s) :
    ∀ᶠ n : ι in p, v < eVariationOn (F n) s := by
  obtain ⟨⟨n, ⟨u, um, us⟩⟩, hlt⟩ :
    ∃ p : ℕ × { u : ℕ → α // Monotone u ∧ ∀ i, u i ∈ s },
      v < ∑ i ∈ Finset.range p.1, edist (f ((p.2 : ℕ → α) (i + 1))) (f ((p.2 : ℕ → α) i)) :=
    lt_iSup_iff.mp hv
  have : Tendsto (fun j => ∑ i ∈ Finset.range n, edist (F j (u (i + 1))) (F j (u i))) p
      (𝓝 (∑ i ∈ Finset.range n, edist (f (u (i + 1))) (f (u i)))) := by
    apply tendsto_finset_sum
    exact fun i _ => Tendsto.edist (Ffs (u i.succ) (us i.succ)) (Ffs (u i) (us i))
  /-
    case intro.mk.mk.intro
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    ι : Type u_3
    F : ι → α → E
    p : Filter ι
    f : α → E
    s : Set α
    Ffs : ∀ (x : α), Membership.mem s x → Filter.Tendsto (fun i => F i x) p (nhds  …
    v : ENNReal
    hv : LT.lt v (eVariationOn f s)
    n : Nat
    u : Nat → α
    um : Monotone u
    us : ∀ (i : Nat), Membership.mem s (u i)
    hlt : LT.lt v ((Finset.range { fst := n, snd := ⟨u, ⋯⟩ }.1).sum fun i => EDist …
    this : Filter.Tendsto (fun j => (Finset.range n).sum fun i => EDist.edist (F j …
    ⊢ Filter.Eventually (fun n => LT.lt v (eVariationOn (F n) s)) p
  -/
  exact (this.eventually_const_lt hlt).mono fun i h => h.trans_le (sum_le (F i) n um us)
  /-
    🎉 no goals
  -/


/-- The map `(eVariationOn · s)` is lower semicontinuous for pointwise convergence *on `s`*.
Pointwise convergence on `s` is encoded here as uniform convergence on the family consisting of the
singletons of elements of `s`.
-/
protected theorem lowerSemicontinuous (s : Set α) :
    LowerSemicontinuous fun f : α →ᵤ[s.image singleton] E => eVariationOn f s := fun f ↦ by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    s : Set α
    f : UniformOnFun α E (Set.image Singleton.singleton s)
    ⊢ LowerSemicontinuousAt (fun f => eVariationOn f s) f
  -/
  apply @lowerSemicontinuous_aux _ _ _ _ (UniformOnFun α E (s.image singleton)) id (𝓝 f) f s _
  simpa only [UniformOnFun.tendsto_iff_tendstoUniformlyOn, mem_image, forall_exists_index, and_imp,
    forall_apply_eq_imp_iff₂, tendstoUniformlyOn_singleton_iff_tendsto] using @tendsto_id _ (𝓝 f)


/-- The map `(eVariationOn · s)` is lower semicontinuous for uniform convergence on `s`. -/
theorem lowerSemicontinuous_uniformOn (s : Set α) :
    LowerSemicontinuous fun f : α →ᵤ[{s}] E => eVariationOn f s := fun f ↦ by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    s : Set α
    f : UniformOnFun α E (Singleton.singleton s)
    ⊢ LowerSemicontinuousAt (fun f => eVariationOn f s) f
  -/
  apply @lowerSemicontinuous_aux _ _ _ _ (UniformOnFun α E {s}) id (𝓝 f) f s _
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    s : Set α
    f : UniformOnFun α E (Singleton.singleton s)
    ⊢ ∀ (x : α), Membership.mem s x → Filter.Tendsto (fun i => id i x) (nhds f) (n …
  -/
  have := @tendsto_id _ (𝓝 f)
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    s : Set α
    f : UniformOnFun α E (Singleton.singleton s)
    this : Filter.Tendsto id (nhds f) (nhds f)
    ⊢ ∀ (x : α), Membership.mem s x → Filter.Tendsto (fun i => id i x) (nhds f) (n …
  -/
  rw [UniformOnFun.tendsto_iff_tendstoUniformlyOn] at this
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    s : Set α
    f : UniformOnFun α E (Singleton.singleton s)
    this : ∀ (s_1 : Set α), Membership.mem (Singleton.singleton s) s_1 → TendstoUn …
    ⊢ ∀ (x : α), Membership.mem s x → Filter.Tendsto (fun i => id i x) (nhds f) (n …
  -/
  simp_rw [← tendstoUniformlyOn_singleton_iff_tendsto]
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    s : Set α
    f : UniformOnFun α E (Singleton.singleton s)
    this : ∀ (s_1 : Set α), Membership.mem (Singleton.singleton s) s_1 → TendstoUn …
    ⊢ ∀ (x : α), Membership.mem s x → TendstoUniformlyOn id f (nhds f) (Singleton. …
  -/
  exact fun x xs => (this s rfl).mono (singleton_subset_iff.mpr xs)
  /-
    🎉 no goals
  -/


theorem _root_.BoundedVariationOn.dist_le {E : Type*} [PseudoMetricSpace E] {f : α → E}
    {s : Set α} (h : BoundedVariationOn f s) {x y : α} (hx : x ∈ s) (hy : y ∈ s) :
    dist (f x) (f y) ≤ (eVariationOn f s).toReal := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_3
    inst✝ : PseudoMetricSpace E
    f : α → E
    s : Set α
    h : BoundedVariationOn f s
    x y : α
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ LE.le (Dist.dist (f x) (f y)) (eVariationOn f s).toReal
  -/
  rw [← ENNReal.ofReal_le_ofReal_iff ENNReal.toReal_nonneg, ENNReal.ofReal_toReal h, ← edist_dist]
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_3
    inst✝ : PseudoMetricSpace E
    f : α → E
    s : Set α
    h : BoundedVariationOn f s
    x y : α
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ LE.le (EDist.edist (f x) (f y)) (eVariationOn f s)
  -/
  exact edist_le f hx hy
  /-
    🎉 no goals
  -/


theorem _root_.BoundedVariationOn.sub_le {f : α → ℝ} {s : Set α} (h : BoundedVariationOn f s)
    {x y : α} (hx : x ∈ s) (hy : y ∈ s) : f x - f y ≤ (eVariationOn f s).toReal := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    f : α → Real
    s : Set α
    h : BoundedVariationOn f s
    x y : α
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ LE.le (HSub.hSub (f x) (f y)) (eVariationOn f s).toReal
  -/
  apply (le_abs_self _).trans
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    f : α → Real
    s : Set α
    h : BoundedVariationOn f s
    x y : α
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ LE.le (abs (HSub.hSub (f x) (f y))) (eVariationOn f s).toReal
  -/
  rw [← Real.dist_eq]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    f : α → Real
    s : Set α
    h : BoundedVariationOn f s
    x y : α
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ LE.le (Dist.dist (f x) (f y)) (eVariationOn f s).toReal
  -/
  exact h.dist_le hx hy
  /-
    🎉 no goals
  -/


/-- Consider a monotone function `u` parameterizing some points of a set `s`. Given `x ∈ s`, then
one can find another monotone function `v` parameterizing the same points as `u`, with `x` added.
In particular, the variation of a function along `u` is bounded by its variation along `v`. -/
theorem add_point (f : α → E) {s : Set α} {x : α} (hx : x ∈ s) (u : ℕ → α) (hu : Monotone u)
    (us : ∀ i, u i ∈ s) (n : ℕ) :
    ∃ (v : ℕ → α) (m : ℕ), Monotone v ∧ (∀ i, v i ∈ s) ∧ x ∈ v '' Iio m ∧
      (∑ i ∈ Finset.range n, edist (f (u (i + 1))) (f (u i))) ≤
        ∑ j ∈ Finset.range m, edist (f (v (j + 1))) (f (v j)) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    x : α
    hx : Membership.mem s x
    u : Nat → α
    hu : Monotone u
    us : ∀ (i : Nat), Membership.mem s (u i)
    n : Nat
    ⊢ Exists fun v => Exists fun m => And (Monotone v) (And (∀ (i : Nat), Membersh …
  -/
  rcases le_or_lt (u n) x with (h | h)
    /-
      case inl
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      x : α
      hx : Membership.mem s x
      u : Nat → α
      hu : Monotone u
      us : ∀ (i : Nat), Membership.mem s (u i)
      n : Nat
      h : LE.le (u n) x
      ⊢ Exists fun v => Exists fun m => And (Monotone v) (And (∀ (i : Nat), Membersh …
    -/
  · let v i := if i ≤ n then u i else x
    have vs : ∀ i, v i ∈ s := fun i ↦ by
      simp only [v]
      split_ifs
      · exact us i
      · exact hx
    have hv : Monotone v := by
      refine monotone_nat_of_le_succ fun i => ?_
      simp only [v]
      rcases lt_trichotomy i n with (hi | rfl | hi)
      · have : i + 1 ≤ n := Nat.succ_le_of_lt hi
        simp only [hi.le, this, if_true]
        exact hu (Nat.le_succ i)
      · simp only [le_refl, if_true, add_le_iff_nonpos_right, Nat.le_zero, Nat.one_ne_zero,
          if_false, h]
      · have A : ¬i ≤ n := hi.not_le
        have B : ¬i + 1 ≤ n := fun h => A (i.le_succ.trans h)
        simp only [A, B, if_false, le_rfl]
    /-
      case inl
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      x : α
      hx : Membership.mem s x
      u : Nat → α
      hu : Monotone u
      us : ∀ (i : Nat), Membership.mem s (u i)
      n : Nat
      h : LE.le (u n) x
      v : Nat → α := fun i => ite (LE.le i n) (u i) x
      vs : ∀ (i : Nat), Membership.mem s (v i)
      hv : Monotone v
      ⊢ Exists fun v => Exists fun m => And (Monotone v) (And (∀ (i : Nat), Membersh …
    -/
    refine ⟨v, n + 2, hv, vs, (mem_image _ _ _).2 ⟨n + 1, ?_, ?_⟩, ?_⟩
      /-
        case inl.refine_1
        α : Type u_1
        inst✝¹ : LinearOrder α
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : α → E
        s : Set α
        x : α
        hx : Membership.mem s x
        u : Nat → α
        hu : Monotone u
        us : ∀ (i : Nat), Membership.mem s (u i)
        n : Nat
        h : LE.le (u n) x
        v : Nat → α := fun i => ite (LE.le i n) (u i) x
        vs : ∀ (i : Nat), Membership.mem s (v i)
        hv : Monotone v
        ⊢ Membership.mem (Set.Iio (HAdd.hAdd n 2)) (HAdd.hAdd n 1)
      -/
    · rw [mem_Iio]; exact Nat.lt_succ_self (n + 1)
                    /-
                      🎉 no goals
                    -/
      /-
        case inl.refine_2
        α : Type u_1
        inst✝¹ : LinearOrder α
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : α → E
        s : Set α
        x : α
        hx : Membership.mem s x
        u : Nat → α
        hu : Monotone u
        us : ∀ (i : Nat), Membership.mem s (u i)
        n : Nat
        h : LE.le (u n) x
        v : Nat → α := fun i => ite (LE.le i n) (u i) x
        vs : ∀ (i : Nat), Membership.mem s (v i)
        hv : Monotone v
        ⊢ Eq (v (HAdd.hAdd n 1)) x
      -/
    · have : ¬n + 1 ≤ n := Nat.not_succ_le_self n
      /-
        case inl.refine_2
        α : Type u_1
        inst✝¹ : LinearOrder α
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : α → E
        s : Set α
        x : α
        hx : Membership.mem s x
        u : Nat → α
        hu : Monotone u
        us : ∀ (i : Nat), Membership.mem s (u i)
        n : Nat
        h : LE.le (u n) x
        v : Nat → α := fun i => ite (LE.le i n) (u i) x
        vs : ∀ (i : Nat), Membership.mem s (v i)
        hv : Monotone v
        this : Not (LE.le (HAdd.hAdd n 1) n)
        ⊢ Eq (v (HAdd.hAdd n 1)) x
      -/
      simp only [v, this, ite_eq_right_iff, IsEmpty.forall_iff]
      /-
        🎉 no goals
      -/
    · calc
        (∑ i ∈ Finset.range n, edist (f (u (i + 1))) (f (u i))) =
            ∑ i ∈ Finset.range n, edist (f (v (i + 1))) (f (v i)) := by
          apply Finset.sum_congr rfl fun i hi => ?_
          simp only [Finset.mem_range] at hi
          have : i + 1 ≤ n := Nat.succ_le_of_lt hi
          simp only [v, hi.le, this, if_true]
        _ ≤ ∑ j ∈ Finset.range (n + 2), edist (f (v (j + 1))) (f (v j)) :=
          Finset.sum_le_sum_of_subset (Finset.range_mono (Nat.le_add_right n 2))
  /-
    case inr
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    x : α
    hx : Membership.mem s x
    u : Nat → α
    hu : Monotone u
    us : ∀ (i : Nat), Membership.mem s (u i)
    n : Nat
    h : LT.lt x (u n)
    ⊢ Exists fun v => Exists fun m => And (Monotone v) (And (∀ (i : Nat), Membersh …
  -/
  have exists_N : ∃ N, N ≤ n ∧ x < u N := ⟨n, le_rfl, h⟩
  /-
    case inr
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    x : α
    hx : Membership.mem s x
    u : Nat → α
    hu : Monotone u
    us : ∀ (i : Nat), Membership.mem s (u i)
    n : Nat
    h : LT.lt x (u n)
    exists_N : Exists fun N => And (LE.le N n) (LT.lt x (u N))
    ⊢ Exists fun v => Exists fun m => And (Monotone v) (And (∀ (i : Nat), Membersh …
  -/
  let N := Nat.find exists_N
  /-
    case inr
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    x : α
    hx : Membership.mem s x
    u : Nat → α
    hu : Monotone u
    us : ∀ (i : Nat), Membership.mem s (u i)
    n : Nat
    h : LT.lt x (u n)
    exists_N : Exists fun N => And (LE.le N n) (LT.lt x (u N))
    N : Nat := Nat.find exists_N
    ⊢ Exists fun v => Exists fun m => And (Monotone v) (And (∀ (i : Nat), Membersh …
  -/
  have hN : N ≤ n ∧ x < u N := Nat.find_spec exists_N
  /-
    case inr
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    x : α
    hx : Membership.mem s x
    u : Nat → α
    hu : Monotone u
    us : ∀ (i : Nat), Membership.mem s (u i)
    n : Nat
    h : LT.lt x (u n)
    exists_N : Exists fun N => And (LE.le N n) (LT.lt x (u N))
    N : Nat := Nat.find exists_N
    hN : And (LE.le N n) (LT.lt x (u N))
    ⊢ Exists fun v => Exists fun m => And (Monotone v) (And (∀ (i : Nat), Membersh …
  -/
  let w : ℕ → α := fun i => if i < N then u i else if i = N then x else u (i - 1)
  have ws : ∀ i, w i ∈ s := by
    dsimp only [w]
    intro i
    split_ifs
    exacts [us _, hx, us _]
  have hw : Monotone w := by
    apply monotone_nat_of_le_succ fun i => ?_
    dsimp only [w]
    rcases lt_trichotomy (i + 1) N with (hi | hi | hi)
    · have : i < N := Nat.lt_of_le_of_lt (Nat.le_succ i) hi
      simp only [hi, this, if_true]
      exact hu (Nat.le_succ _)
    · have A : i < N := hi ▸ i.lt_succ_self
      have B : ¬i + 1 < N := by rw [← hi]; exact fun h => h.ne rfl
      rw [if_pos A, if_neg B, if_pos hi]
      have T := Nat.find_min exists_N A
      push_neg at T
      exact T (A.le.trans hN.1)
    · have A : ¬i < N := (Nat.lt_succ_iff.mp hi).not_lt
      have B : ¬i + 1 < N := hi.not_lt
      have C : ¬i + 1 = N := hi.ne.symm
      have D : i + 1 - 1 = i := Nat.pred_succ i
      rw [if_neg A, if_neg B, if_neg C, D]
      split_ifs
      · exact hN.2.le.trans (hu (le_of_not_lt A))
      · exact hu (Nat.pred_le _)
  /-
    case inr
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    x : α
    hx : Membership.mem s x
    u : Nat → α
    hu : Monotone u
    us : ∀ (i : Nat), Membership.mem s (u i)
    n : Nat
    h : LT.lt x (u n)
    exists_N : Exists fun N => And (LE.le N n) (LT.lt x (u N))
    N : Nat := Nat.find exists_N
    hN : And (LE.le N n) (LT.lt x (u N))
    w : Nat → α := fun i => ite (LT.lt i N) (u i) (ite (Eq i N) x (u (HSub.hSub i  …
    ws : ∀ (i : Nat), Membership.mem s (w i)
    hw : Monotone w
    ⊢ Exists fun v => Exists fun m => And (Monotone v) (And (∀ (i : Nat), Membersh …
  -/
  refine ⟨w, n + 1, hw, ws, (mem_image _ _ _).2 ⟨N, hN.1.trans_lt (Nat.lt_succ_self n), ?_⟩, ?_⟩
    /-
      case inr.refine_1
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      x : α
      hx : Membership.mem s x
      u : Nat → α
      hu : Monotone u
      us : ∀ (i : Nat), Membership.mem s (u i)
      n : Nat
      h : LT.lt x (u n)
      exists_N : Exists fun N => And (LE.le N n) (LT.lt x (u N))
      N : Nat := Nat.find exists_N
      hN : And (LE.le N n) (LT.lt x (u N))
      w : Nat → α := fun i => ite (LT.lt i N) (u i) (ite (Eq i N) x (u (HSub.hSub i  …
      ws : ∀ (i : Nat), Membership.mem s (w i)
      hw : Monotone w
      ⊢ Eq (w N) x
    -/
  · dsimp only [w]; rw [if_neg (lt_irrefl N), if_pos rfl]
                    /-
                      🎉 no goals
                    -/
  /-
    case inr.refine_2
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    x : α
    hx : Membership.mem s x
    u : Nat → α
    hu : Monotone u
    us : ∀ (i : Nat), Membership.mem s (u i)
    n : Nat
    h : LT.lt x (u n)
    exists_N : Exists fun N => And (LE.le N n) (LT.lt x (u N))
    N : Nat := Nat.find exists_N
    hN : And (LE.le N n) (LT.lt x (u N))
    w : Nat → α := fun i => ite (LT.lt i N) (u i) (ite (Eq i N) x (u (HSub.hSub i  …
    ws : ∀ (i : Nat), Membership.mem s (w i)
    hw : Monotone w
    ⊢ LE.le ((Finset.range n).sum fun i => EDist.edist (f (u (HAdd.hAdd i 1))) (f  …
  -/
  rcases eq_or_lt_of_le (zero_le N) with (Npos | Npos)
  · calc
      (∑ i ∈ Finset.range n, edist (f (u (i + 1))) (f (u i))) =
          ∑ i ∈ Finset.range n, edist (f (w (1 + i + 1))) (f (w (1 + i))) := by
        apply Finset.sum_congr rfl fun i _hi => ?_
        dsimp only [w]
        simp only [← Npos, Nat.not_lt_zero, Nat.add_succ_sub_one, add_zero, if_false,
          add_eq_zero, Nat.one_ne_zero, false_and, Nat.succ_add_sub_one, zero_add]
        rw [add_comm 1 i]
      _ = ∑ i ∈ Finset.Ico 1 (n + 1), edist (f (w (i + 1))) (f (w i)) := by
        rw [Finset.range_eq_Ico]
        exact Finset.sum_Ico_add (fun i => edist (f (w (i + 1))) (f (w i))) 0 n 1
      _ ≤ ∑ j ∈ Finset.range (n + 1), edist (f (w (j + 1))) (f (w j)) := by
        apply Finset.sum_le_sum_of_subset _
        rw [Finset.range_eq_Ico]
        exact Finset.Ico_subset_Ico zero_le_one le_rfl
  · calc
      (∑ i ∈ Finset.range n, edist (f (u (i + 1))) (f (u i))) =
          ((∑ i ∈ Finset.Ico 0 (N - 1), edist (f (u (i + 1))) (f (u i))) +
              ∑ i ∈ Finset.Ico (N - 1) N, edist (f (u (i + 1))) (f (u i))) +
            ∑ i ∈ Finset.Ico N n, edist (f (u (i + 1))) (f (u i)) := by
        rw [Finset.sum_Ico_consecutive, Finset.sum_Ico_consecutive, Finset.range_eq_Ico]
        · exact zero_le _
        · exact hN.1
        · exact zero_le _
        · exact Nat.pred_le _
      _ = (∑ i ∈ Finset.Ico 0 (N - 1), edist (f (w (i + 1))) (f (w i))) +
              edist (f (u N)) (f (u (N - 1))) +
            ∑ i ∈ Finset.Ico N n, edist (f (w (1 + i + 1))) (f (w (1 + i))) := by
        congr 1
        · congr 1
          · apply Finset.sum_congr rfl fun i hi => ?_
            simp only [Finset.mem_Ico, zero_le', true_and] at hi
            dsimp only [w]
            have A : i + 1 < N := Nat.lt_pred_iff.1 hi
            have B : i < N := Nat.lt_of_succ_lt A
            rw [if_pos A, if_pos B]
          · have A : N - 1 + 1 = N := Nat.succ_pred_eq_of_pos Npos
            have : Finset.Ico (N - 1) N = {N - 1} := by rw [← Nat.Ico_succ_singleton, A]
            simp only [this, A, Finset.sum_singleton]
        · apply Finset.sum_congr rfl fun i hi => ?_
          rw [Finset.mem_Ico] at hi
          dsimp only [w]
          have A : ¬1 + i + 1 < N := by omega
          have B : ¬1 + i + 1 = N := by omega
          have C : ¬1 + i < N := by omega
          have D : ¬1 + i = N := by omega
          rw [if_neg A, if_neg B, if_neg C, if_neg D]
          congr 3 <;> · rw [add_comm, Nat.sub_one]; apply Nat.pred_succ
      _ = (∑ i ∈ Finset.Ico 0 (N - 1), edist (f (w (i + 1))) (f (w i))) +
              edist (f (w (N + 1))) (f (w (N - 1))) +
            ∑ i ∈ Finset.Ico (N + 1) (n + 1), edist (f (w (i + 1))) (f (w i)) := by
        congr 1
        · congr 1
          · dsimp only [w]
            have A : ¬N + 1 < N := Nat.not_succ_lt_self
            have B : N - 1 < N := Nat.pred_lt Npos.ne'
            simp only [A, not_and, not_lt, Nat.succ_ne_self, Nat.add_succ_sub_one, add_zero,
              if_false, B, if_true]
        · exact Finset.sum_Ico_add (fun i => edist (f (w (i + 1))) (f (w i))) N n 1
      _ ≤ ((∑ i ∈ Finset.Ico 0 (N - 1), edist (f (w (i + 1))) (f (w i))) +
              ∑ i ∈ Finset.Ico (N - 1) (N + 1), edist (f (w (i + 1))) (f (w i))) +
            ∑ i ∈ Finset.Ico (N + 1) (n + 1), edist (f (w (i + 1))) (f (w i)) := by
        refine add_le_add (add_le_add le_rfl ?_) le_rfl
        have A : N - 1 + 1 = N := Nat.succ_pred_eq_of_pos Npos
        have B : N - 1 + 1 < N + 1 := A.symm ▸ N.lt_succ_self
        have C : N - 1 < N + 1 := lt_of_le_of_lt N.pred_le N.lt_succ_self
        rw [Finset.sum_eq_sum_Ico_succ_bot C, Finset.sum_eq_sum_Ico_succ_bot B, A, Finset.Ico_self,
          Finset.sum_empty, add_zero, add_comm (edist _ _)]
        exact edist_triangle _ _ _
      _ = ∑ j ∈ Finset.range (n + 1), edist (f (w (j + 1))) (f (w j)) := by
        rw [Finset.sum_Ico_consecutive, Finset.sum_Ico_consecutive, Finset.range_eq_Ico]
        · exact zero_le _
        · exact Nat.succ_le_succ hN.left
        · exact zero_le _
        · exact N.pred_le.trans N.le_succ


/-- The variation of a function on the union of two sets `s` and `t`, with `s` to the left of `t`,
bounds the sum of the variations along `s` and `t`. -/
theorem add_le_union (f : α → E) {s t : Set α} (h : ∀ x ∈ s, ∀ y ∈ t, x ≤ y) :
    eVariationOn f s + eVariationOn f t ≤ eVariationOn f (s ∪ t) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s t : Set α
    h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem t y → LE.le x y
    ⊢ LE.le (HAdd.hAdd (eVariationOn f s) (eVariationOn f t)) (eVariationOn f (Uni …
  -/
  by_cases hs : s = ∅
    /-
      case pos
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s t : Set α
      h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem t y → LE.le x y
      hs : Eq s EmptyCollection.emptyCollection
      ⊢ LE.le (HAdd.hAdd (eVariationOn f s) (eVariationOn f t)) (eVariationOn f (Uni …
    -/
  · simp [hs]
    /-
      🎉 no goals
    -/
  have : Nonempty { u // Monotone u ∧ ∀ i : ℕ, u i ∈ s } :=
    nonempty_monotone_mem (nonempty_iff_ne_empty.2 hs)
  /-
    case neg
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s t : Set α
    h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem t y → LE.le x y
    hs : Not (Eq s EmptyCollection.emptyCollection)
    this : Nonempty (Subtype fun u => And (Monotone u) (∀ (i : Nat), Membership.me …
    ⊢ LE.le (HAdd.hAdd (eVariationOn f s) (eVariationOn f t)) (eVariationOn f (Uni …
  -/
  by_cases ht : t = ∅
    /-
      case pos
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s t : Set α
      h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem t y → LE.le x y
      hs : Not (Eq s EmptyCollection.emptyCollection)
      this : Nonempty (Subtype fun u => And (Monotone u) (∀ (i : Nat), Membership.me …
      ht : Eq t EmptyCollection.emptyCollection
      ⊢ LE.le (HAdd.hAdd (eVariationOn f s) (eVariationOn f t)) (eVariationOn f (Uni …
    -/
  · simp [ht]
    /-
      🎉 no goals
    -/
  have : Nonempty { u // Monotone u ∧ ∀ i : ℕ, u i ∈ t } :=
    nonempty_monotone_mem (nonempty_iff_ne_empty.2 ht)
  /-
    case neg
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s t : Set α
    h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem t y → LE.le x y
    hs : Not (Eq s EmptyCollection.emptyCollection)
    this✝ : Nonempty (Subtype fun u => And (Monotone u) (∀ (i : Nat), Membership.m …
    ht : Not (Eq t EmptyCollection.emptyCollection)
    this : Nonempty (Subtype fun u => And (Monotone u) (∀ (i : Nat), Membership.me …
    ⊢ LE.le (HAdd.hAdd (eVariationOn f s) (eVariationOn f t)) (eVariationOn f (Uni …
  -/
  refine ENNReal.iSup_add_iSup_le ?_
  /- We start from two sequences `u` and `v` along `s` and `t` respectively, and we build a new
    sequence `w` along `s ∪ t` by juxtaposing them. Its variation is larger than the sum of the
    variations. -/
  /-
    case neg
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s t : Set α
    h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem t y → LE.le x y
    hs : Not (Eq s EmptyCollection.emptyCollection)
    this✝ : Nonempty (Subtype fun u => And (Monotone u) (∀ (i : Nat), Membership.m …
    ht : Not (Eq t EmptyCollection.emptyCollection)
    this : Nonempty (Subtype fun u => And (Monotone u) (∀ (i : Nat), Membership.me …
    ⊢ ∀ (i : Prod Nat (Subtype fun u => And (Monotone u) (∀ (i : Nat), Membership. …
  -/
  rintro ⟨n, ⟨u, hu, us⟩⟩ ⟨m, ⟨v, hv, vt⟩⟩
  /-
    case neg.mk.mk.intro.mk.mk.intro
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s t : Set α
    h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem t y → LE.le x y
    hs : Not (Eq s EmptyCollection.emptyCollection)
    this✝ : Nonempty (Subtype fun u => And (Monotone u) (∀ (i : Nat), Membership.m …
    ht : Not (Eq t EmptyCollection.emptyCollection)
    this : Nonempty (Subtype fun u => And (Monotone u) (∀ (i : Nat), Membership.me …
    n : Nat
    u : Nat → α
    hu : Monotone u
    us : ∀ (i : Nat), Membership.mem s (u i)
    m : Nat
    v : Nat → α
    hv : Monotone v
    vt : ∀ (i : Nat), Membership.mem t (v i)
    ⊢ LE.le (HAdd.hAdd ((Finset.range { fst := n, snd := ⟨u, ⋯⟩ }.1).sum fun i =>  …
  -/
  let w i := if i ≤ n then u i else v (i - (n + 1))
  have wst : ∀ i, w i ∈ s ∪ t := by
    intro i
    by_cases hi : i ≤ n
    · simp [w, hi, us]
    · simp [w, hi, vt]
  have hw : Monotone w := by
    intro i j hij
    dsimp only [w]
    split_ifs with h_1 h_2 h_2
    · exact hu hij
    · apply h _ (us _) _ (vt _)
    · exfalso; exact h_1 (hij.trans h_2)
    · apply hv (tsub_le_tsub hij le_rfl)
  calc
    ((∑ i ∈ Finset.range n, edist (f (u (i + 1))) (f (u i))) +
          ∑ i ∈ Finset.range m, edist (f (v (i + 1))) (f (v i))) =
        (∑ i ∈ Finset.range n, edist (f (w (i + 1))) (f (w i))) +
          ∑ i ∈ Finset.range m, edist (f (w (n + 1 + i + 1))) (f (w (n + 1 + i))) := by
      dsimp only [w]
      congr 1
      · refine Finset.sum_congr rfl fun i hi => ?_
        simp only [Finset.mem_range] at hi
        have : i + 1 ≤ n := Nat.succ_le_of_lt hi
        simp [hi.le, this]
      · refine Finset.sum_congr rfl fun i hi => ?_
        simp only [Finset.mem_range] at hi
        have B : ¬n + 1 + i ≤ n := by omega
        have A : ¬n + 1 + i + 1 ≤ n := fun h => B ((n + 1 + i).le_succ.trans h)
        have C : n + 1 + i - n = i + 1 := by
          rw [tsub_eq_iff_eq_add_of_le]
          · abel
          · exact n.le_succ.trans (n.succ.le_add_right i)
        simp only [A, B, C, Nat.succ_sub_succ_eq_sub, if_false, add_tsub_cancel_left]
    _ = (∑ i ∈ Finset.range n, edist (f (w (i + 1))) (f (w i))) +
          ∑ i ∈ Finset.Ico (n + 1) (n + 1 + m), edist (f (w (i + 1))) (f (w i)) := by
      congr 1
      rw [Finset.range_eq_Ico]
      convert Finset.sum_Ico_add (fun i : ℕ => edist (f (w (i + 1))) (f (w i))) 0 m (n + 1)
        using 3 <;> abel
    _ ≤ ∑ i ∈ Finset.range (n + 1 + m), edist (f (w (i + 1))) (f (w i)) := by
      rw [← Finset.sum_union]
      · apply Finset.sum_le_sum_of_subset _
        rintro i hi
        simp only [Finset.mem_union, Finset.mem_range, Finset.mem_Ico] at hi ⊢
        cases' hi with hi hi
        · exact lt_of_lt_of_le hi (n.le_succ.trans (n.succ.le_add_right m))
        · exact hi.2
      · refine Finset.disjoint_left.2 fun i hi h'i => ?_
        simp only [Finset.mem_Ico, Finset.mem_range] at hi h'i
        exact hi.not_lt (Nat.lt_of_succ_le h'i.left)
    _ ≤ eVariationOn f (s ∪ t) := sum_le f _ hw wst


/-- If a set `s` is to the left of a set `t`, and both contain the boundary point `x`, then
the variation of `f` along `s ∪ t` is the sum of the variations. -/
theorem union (f : α → E) {s t : Set α} {x : α} (hs : IsGreatest s x) (ht : IsLeast t x) :
    eVariationOn f (s ∪ t) = eVariationOn f s + eVariationOn f t := by
  classical
  apply le_antisymm _ (eVariationOn.add_le_union f fun a ha b hb => le_trans (hs.2 ha) (ht.2 hb))
  apply iSup_le _
  rintro ⟨n, ⟨u, hu, ust⟩⟩
  obtain ⟨v, m, hv, vst, xv, huv⟩ : ∃ (v : ℕ → α) (m : ℕ),
    Monotone v ∧ (∀ i, v i ∈ s ∪ t) ∧ x ∈ v '' Iio m ∧
      (∑ i ∈ Finset.range n, edist (f (u (i + 1))) (f (u i))) ≤
        ∑ j ∈ Finset.range m, edist (f (v (j + 1))) (f (v j)) :=
    eVariationOn.add_point f (mem_union_left t hs.1) u hu ust n
  obtain ⟨N, hN, Nx⟩ : ∃ N, N < m ∧ v N = x := xv
  calc
    (∑ j ∈ Finset.range n, edist (f (u (j + 1))) (f (u j))) ≤
        ∑ j ∈ Finset.range m, edist (f (v (j + 1))) (f (v j)) :=
      huv
    _ = (∑ j ∈ Finset.Ico 0 N, edist (f (v (j + 1))) (f (v j))) +
          ∑ j ∈ Finset.Ico N m, edist (f (v (j + 1))) (f (v j)) := by
      rw [Finset.range_eq_Ico, Finset.sum_Ico_consecutive _ (zero_le _) hN.le]
    _ ≤ eVariationOn f s + eVariationOn f t := by
      refine add_le_add ?_ ?_
      · apply sum_le_of_monotoneOn_Icc _ (hv.monotoneOn _) fun i hi => ?_
        rcases vst i with (h | h); · exact h
        have : v i = x := by
          apply le_antisymm
          · rw [← Nx]; exact hv hi.2
          · exact ht.2 h
        rw [this]
        exact hs.1
      · apply sum_le_of_monotoneOn_Icc _ (hv.monotoneOn _) fun i hi => ?_
        rcases vst i with (h | h); swap; · exact h
        have : v i = x := by
          apply le_antisymm
          · exact hs.2 h
          · rw [← Nx]; exact hv hi.1
        rw [this]
        exact ht.1


theorem Icc_add_Icc (f : α → E) {s : Set α} {a b c : α} (hab : a ≤ b) (hbc : b ≤ c) (hb : b ∈ s) :
    eVariationOn f (s ∩ Icc a b) + eVariationOn f (s ∩ Icc b c) = eVariationOn f (s ∩ Icc a c) := by
  have A : IsGreatest (s ∩ Icc a b) b :=
    ⟨⟨hb, hab, le_rfl⟩, inter_subset_right.trans Icc_subset_Iic_self⟩
  have B : IsLeast (s ∩ Icc b c) b :=
    ⟨⟨hb, le_rfl, hbc⟩, inter_subset_right.trans Icc_subset_Ici_self⟩
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    a b c : α
    hab : LE.le a b
    hbc : LE.le b c
    hb : Membership.mem s b
    A : IsGreatest (Inter.inter s (Set.Icc a b)) b
    B : IsLeast (Inter.inter s (Set.Icc b c)) b
    ⊢ Eq (HAdd.hAdd (eVariationOn f (Inter.inter s (Set.Icc a b))) (eVariationOn f …
  -/
  rw [← eVariationOn.union f A B, ← inter_union_distrib_left, Icc_union_Icc_eq_Icc hab hbc]
  /-
    🎉 no goals
  -/


theorem comp_le_of_monotoneOn (f : α → E) {s : Set α} {t : Set β} (φ : β → α) (hφ : MonotoneOn φ t)
    (φst : MapsTo φ t s) : eVariationOn (f ∘ φ) t ≤ eVariationOn f s :=
  iSup_le fun ⟨n, u, hu, ut⟩ =>
    le_iSup_of_le ⟨n, φ ∘ u, fun x y xy => hφ (ut x) (ut y) (hu xy), fun i => φst (ut i)⟩ le_rfl


theorem comp_le_of_antitoneOn (f : α → E) {s : Set α} {t : Set β} (φ : β → α) (hφ : AntitoneOn φ t)
    (φst : MapsTo φ t s) : eVariationOn (f ∘ φ) t ≤ eVariationOn f s := by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    s : Set α
    t : Set β
    φ : β → α
    hφ : AntitoneOn φ t
    φst : Set.MapsTo φ t s
    ⊢ LE.le (eVariationOn (Function.comp f φ) t) (eVariationOn f s)
  -/
  refine iSup_le ?_
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    s : Set α
    t : Set β
    φ : β → α
    hφ : AntitoneOn φ t
    φst : Set.MapsTo φ t s
    ⊢ ∀ (i : Prod Nat (Subtype fun u => And (Monotone u) (∀ (i : Nat), Membership. …
  -/
  rintro ⟨n, u, hu, ut⟩
  /-
    case mk.mk.intro
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    s : Set α
    t : Set β
    φ : β → α
    hφ : AntitoneOn φ t
    φst : Set.MapsTo φ t s
    n : Nat
    u : Nat → β
    hu : Monotone u
    ut : ∀ (i : Nat), Membership.mem t (u i)
    ⊢ LE.le ((Finset.range { fst := n, snd := ⟨u, ⋯⟩ }.1).sum fun i => EDist.edist …
  -/
  rw [← Finset.sum_range_reflect]
  refine (Finset.sum_congr rfl fun x hx => ?_).trans_le <| le_iSup_of_le
    ⟨n, fun i => φ (u <| n - i), fun x y xy => hφ (ut _) (ut _) (hu <| Nat.sub_le_sub_left xy n),
      fun i => φst (ut _)⟩
    le_rfl
  /-
    case mk.mk.intro
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    s : Set α
    t : Set β
    φ : β → α
    hφ : AntitoneOn φ t
    φst : Set.MapsTo φ t s
    n : Nat
    u : Nat → β
    hu : Monotone u
    ut : ∀ (i : Nat), Membership.mem t (u i)
    x : Nat
    hx : Membership.mem (Finset.range { fst := n, snd := ⟨u, ⋯⟩ }.1) x
    ⊢ Eq (EDist.edist (Function.comp f φ (↑{ fst := n, snd := ⟨u, ⋯⟩ }.2 (HAdd.hAd …
  -/
  rw [Finset.mem_range] at hx
  /-
    case mk.mk.intro
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    s : Set α
    t : Set β
    φ : β → α
    hφ : AntitoneOn φ t
    φst : Set.MapsTo φ t s
    n : Nat
    u : Nat → β
    hu : Monotone u
    ut : ∀ (i : Nat), Membership.mem t (u i)
    x : Nat
    hx : LT.lt x { fst := n, snd := ⟨u, ⋯⟩ }.1
    ⊢ Eq (EDist.edist (Function.comp f φ (↑{ fst := n, snd := ⟨u, ⋯⟩ }.2 (HAdd.hAd …
  -/
  dsimp only [Subtype.coe_mk, Function.comp_apply]
  /-
    case mk.mk.intro
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    s : Set α
    t : Set β
    φ : β → α
    hφ : AntitoneOn φ t
    φst : Set.MapsTo φ t s
    n : Nat
    u : Nat → β
    hu : Monotone u
    ut : ∀ (i : Nat), Membership.mem t (u i)
    x : Nat
    hx : LT.lt x { fst := n, snd := ⟨u, ⋯⟩ }.1
    ⊢ Eq (EDist.edist (f (φ (u (HAdd.hAdd (HSub.hSub (HSub.hSub n 1) x) 1)))) (f ( …
  -/
  rw [edist_comm]
  /-
    case mk.mk.intro
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    s : Set α
    t : Set β
    φ : β → α
    hφ : AntitoneOn φ t
    φst : Set.MapsTo φ t s
    n : Nat
    u : Nat → β
    hu : Monotone u
    ut : ∀ (i : Nat), Membership.mem t (u i)
    x : Nat
    hx : LT.lt x { fst := n, snd := ⟨u, ⋯⟩ }.1
    ⊢ Eq (EDist.edist (f (φ (u (HSub.hSub (HSub.hSub n 1) x)))) (f (φ (u (HAdd.hAd …
  -/
              /-
                🎉 no goals
              -/
  congr 4 <;> omega
              /-
                🎉 no goals
              -/


theorem comp_eq_of_monotoneOn (f : α → E) {t : Set β} (φ : β → α) (hφ : MonotoneOn φ t) :
    eVariationOn (f ∘ φ) t = eVariationOn f (φ '' t) := by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : MonotoneOn φ t
    ⊢ Eq (eVariationOn (Function.comp f φ) t) (eVariationOn f (Set.image φ t))
  -/
  apply le_antisymm (comp_le_of_monotoneOn f φ hφ (mapsTo_image φ t))
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : MonotoneOn φ t
    ⊢ LE.le (eVariationOn f (Set.image φ t)) (eVariationOn (Function.comp f φ) t)
  -/
  cases isEmpty_or_nonempty β
    /-
      case inl
      α : Type u_1
      inst✝² : LinearOrder α
      E : Type u_2
      inst✝¹ : PseudoEMetricSpace E
      β : Type u_3
      inst✝ : LinearOrder β
      f : α → E
      t : Set β
      φ : β → α
      hφ : MonotoneOn φ t
      h✝ : IsEmpty β
      ⊢ LE.le (eVariationOn f (Set.image φ t)) (eVariationOn (Function.comp f φ) t)
    -/
  · convert zero_le (_ : ℝ≥0∞)
    exact eVariationOn.subsingleton f <|
      (subsingleton_of_subsingleton.image _).anti (surjOn_image φ t)
  /-
    case inr
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : MonotoneOn φ t
    h✝ : Nonempty β
    ⊢ LE.le (eVariationOn f (Set.image φ t)) (eVariationOn (Function.comp f φ) t)
  -/
  let ψ := φ.invFunOn t
  /-
    case inr
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : MonotoneOn φ t
    h✝ : Nonempty β
    ψ : α → β := Function.invFunOn φ t
    ⊢ LE.le (eVariationOn f (Set.image φ t)) (eVariationOn (Function.comp f φ) t)
  -/
  have ψφs : EqOn (φ ∘ ψ) id (φ '' t) := (surjOn_image φ t).rightInvOn_invFunOn
  /-
    case inr
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : MonotoneOn φ t
    h✝ : Nonempty β
    ψ : α → β := Function.invFunOn φ t
    ψφs : Set.EqOn (Function.comp φ ψ) id (Set.image φ t)
    ⊢ LE.le (eVariationOn f (Set.image φ t)) (eVariationOn (Function.comp f φ) t)
  -/
  have ψts : MapsTo ψ (φ '' t) t := (surjOn_image φ t).mapsTo_invFunOn
  /-
    case inr
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : MonotoneOn φ t
    h✝ : Nonempty β
    ψ : α → β := Function.invFunOn φ t
    ψφs : Set.EqOn (Function.comp φ ψ) id (Set.image φ t)
    ψts : Set.MapsTo ψ (Set.image φ t) t
    ⊢ LE.le (eVariationOn f (Set.image φ t)) (eVariationOn (Function.comp f φ) t)
  -/
  have hψ : MonotoneOn ψ (φ '' t) := Function.monotoneOn_of_rightInvOn_of_mapsTo hφ ψφs ψts
  /-
    case inr
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : MonotoneOn φ t
    h✝ : Nonempty β
    ψ : α → β := Function.invFunOn φ t
    ψφs : Set.EqOn (Function.comp φ ψ) id (Set.image φ t)
    ψts : Set.MapsTo ψ (Set.image φ t) t
    hψ : MonotoneOn ψ (Set.image φ t)
    ⊢ LE.le (eVariationOn f (Set.image φ t)) (eVariationOn (Function.comp f φ) t)
  -/
  change eVariationOn (f ∘ id) (φ '' t) ≤ eVariationOn (f ∘ φ) t
  /-
    case inr
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : MonotoneOn φ t
    h✝ : Nonempty β
    ψ : α → β := Function.invFunOn φ t
    ψφs : Set.EqOn (Function.comp φ ψ) id (Set.image φ t)
    ψts : Set.MapsTo ψ (Set.image φ t) t
    hψ : MonotoneOn ψ (Set.image φ t)
    ⊢ LE.le (eVariationOn (Function.comp f id) (Set.image φ t)) (eVariationOn (Fun …
  -/
  rw [← eq_of_eqOn (ψφs.comp_left : EqOn (f ∘ φ ∘ ψ) (f ∘ id) (φ '' t))]
  /-
    case inr
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : MonotoneOn φ t
    h✝ : Nonempty β
    ψ : α → β := Function.invFunOn φ t
    ψφs : Set.EqOn (Function.comp φ ψ) id (Set.image φ t)
    ψts : Set.MapsTo ψ (Set.image φ t) t
    hψ : MonotoneOn ψ (Set.image φ t)
    ⊢ LE.le (eVariationOn (Function.comp f (Function.comp φ ψ)) (Set.image φ t)) ( …
  -/
  exact comp_le_of_monotoneOn _ ψ hψ ψts
  /-
    🎉 no goals
  -/


theorem comp_inter_Icc_eq_of_monotoneOn (f : α → E) {t : Set β} (φ : β → α) (hφ : MonotoneOn φ t)
    {x y : β} (hx : x ∈ t) (hy : y ∈ t) :
    eVariationOn (f ∘ φ) (t ∩ Icc x y) = eVariationOn f (φ '' t ∩ Icc (φ x) (φ y)) := by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : MonotoneOn φ t
    x y : β
    hx : Membership.mem t x
    hy : Membership.mem t y
    ⊢ Eq (eVariationOn (Function.comp f φ) (Inter.inter t (Set.Icc x y))) (eVariat …
  -/
  rcases le_total x y with (h | h)
    /-
      case inl
      α : Type u_1
      inst✝² : LinearOrder α
      E : Type u_2
      inst✝¹ : PseudoEMetricSpace E
      β : Type u_3
      inst✝ : LinearOrder β
      f : α → E
      t : Set β
      φ : β → α
      hφ : MonotoneOn φ t
      x y : β
      hx : Membership.mem t x
      hy : Membership.mem t y
      h : LE.le x y
      ⊢ Eq (eVariationOn (Function.comp f φ) (Inter.inter t (Set.Icc x y))) (eVariat …
    -/
  · convert comp_eq_of_monotoneOn f φ (hφ.mono Set.inter_subset_left)
    /-
      case h.e'_3.h.e'_6
      α : Type u_1
      inst✝² : LinearOrder α
      E : Type u_2
      inst✝¹ : PseudoEMetricSpace E
      β : Type u_3
      inst✝ : LinearOrder β
      f : α → E
      t : Set β
      φ : β → α
      hφ : MonotoneOn φ t
      x y : β
      hx : Membership.mem t x
      hy : Membership.mem t y
      h : LE.le x y
      ⊢ Eq (Inter.inter (Set.image φ t) (Set.Icc (φ x) (φ y))) (Set.image φ (Inter.i …
    -/
    apply le_antisymm
      /-
        case h.e'_3.h.e'_6.a
        α : Type u_1
        inst✝² : LinearOrder α
        E : Type u_2
        inst✝¹ : PseudoEMetricSpace E
        β : Type u_3
        inst✝ : LinearOrder β
        f : α → E
        t : Set β
        φ : β → α
        hφ : MonotoneOn φ t
        x y : β
        hx : Membership.mem t x
        hy : Membership.mem t y
        h : LE.le x y
        ⊢ LE.le (Inter.inter (Set.image φ t) (Set.Icc (φ x) (φ y))) (Set.image φ (Inte …
      -/
    · rintro _ ⟨⟨u, us, rfl⟩, vφx, vφy⟩
      /-
        case h.e'_3.h.e'_6.a.intro.intro.intro.intro
        α : Type u_1
        inst✝² : LinearOrder α
        E : Type u_2
        inst✝¹ : PseudoEMetricSpace E
        β : Type u_3
        inst✝ : LinearOrder β
        f : α → E
        t : Set β
        φ : β → α
        hφ : MonotoneOn φ t
        x y : β
        hx : Membership.mem t x
        hy : Membership.mem t y
        h : LE.le x y
        u : β
        us : Membership.mem t u
        vφx : LE.le (φ x) (φ u)
        vφy : LE.le (φ u) (φ y)
        ⊢ Membership.mem (Set.image φ (Inter.inter t (Set.Icc x y))) (φ u)
      -/
      rcases le_total x u with (xu | ux)
        /-
          case h.e'_3.h.e'_6.a.intro.intro.intro.intro.inl
          α : Type u_1
          inst✝² : LinearOrder α
          E : Type u_2
          inst✝¹ : PseudoEMetricSpace E
          β : Type u_3
          inst✝ : LinearOrder β
          f : α → E
          t : Set β
          φ : β → α
          hφ : MonotoneOn φ t
          x y : β
          hx : Membership.mem t x
          hy : Membership.mem t y
          h : LE.le x y
          u : β
          us : Membership.mem t u
          vφx : LE.le (φ x) (φ u)
          vφy : LE.le (φ u) (φ y)
          xu : LE.le x u
          ⊢ Membership.mem (Set.image φ (Inter.inter t (Set.Icc x y))) (φ u)
        -/
      · rcases le_total u y with (uy | yu)
          /-
            case h.e'_3.h.e'_6.a.intro.intro.intro.intro.inl.inl
            α : Type u_1
            inst✝² : LinearOrder α
            E : Type u_2
            inst✝¹ : PseudoEMetricSpace E
            β : Type u_3
            inst✝ : LinearOrder β
            f : α → E
            t : Set β
            φ : β → α
            hφ : MonotoneOn φ t
            x y : β
            hx : Membership.mem t x
            hy : Membership.mem t y
            h : LE.le x y
            u : β
            us : Membership.mem t u
            vφx : LE.le (φ x) (φ u)
            vφy : LE.le (φ u) (φ y)
            xu : LE.le x u
            uy : LE.le u y
            ⊢ Membership.mem (Set.image φ (Inter.inter t (Set.Icc x y))) (φ u)
          -/
        · exact ⟨u, ⟨us, ⟨xu, uy⟩⟩, rfl⟩
          /-
            🎉 no goals
          -/
          /-
            case h.e'_3.h.e'_6.a.intro.intro.intro.intro.inl.inr
            α : Type u_1
            inst✝² : LinearOrder α
            E : Type u_2
            inst✝¹ : PseudoEMetricSpace E
            β : Type u_3
            inst✝ : LinearOrder β
            f : α → E
            t : Set β
            φ : β → α
            hφ : MonotoneOn φ t
            x y : β
            hx : Membership.mem t x
            hy : Membership.mem t y
            h : LE.le x y
            u : β
            us : Membership.mem t u
            vφx : LE.le (φ x) (φ u)
            vφy : LE.le (φ u) (φ y)
            xu : LE.le x u
            yu : LE.le y u
            ⊢ Membership.mem (Set.image φ (Inter.inter t (Set.Icc x y))) (φ u)
          -/
        · rw [le_antisymm vφy (hφ hy us yu)]
          /-
            case h.e'_3.h.e'_6.a.intro.intro.intro.intro.inl.inr
            α : Type u_1
            inst✝² : LinearOrder α
            E : Type u_2
            inst✝¹ : PseudoEMetricSpace E
            β : Type u_3
            inst✝ : LinearOrder β
            f : α → E
            t : Set β
            φ : β → α
            hφ : MonotoneOn φ t
            x y : β
            hx : Membership.mem t x
            hy : Membership.mem t y
            h : LE.le x y
            u : β
            us : Membership.mem t u
            vφx : LE.le (φ x) (φ u)
            vφy : LE.le (φ u) (φ y)
            xu : LE.le x u
            yu : LE.le y u
            ⊢ Membership.mem (Set.image φ (Inter.inter t (Set.Icc x y))) (φ y)
          -/
          exact ⟨y, ⟨hy, ⟨h, le_rfl⟩⟩, rfl⟩
          /-
            🎉 no goals
          -/
        /-
          case h.e'_3.h.e'_6.a.intro.intro.intro.intro.inr
          α : Type u_1
          inst✝² : LinearOrder α
          E : Type u_2
          inst✝¹ : PseudoEMetricSpace E
          β : Type u_3
          inst✝ : LinearOrder β
          f : α → E
          t : Set β
          φ : β → α
          hφ : MonotoneOn φ t
          x y : β
          hx : Membership.mem t x
          hy : Membership.mem t y
          h : LE.le x y
          u : β
          us : Membership.mem t u
          vφx : LE.le (φ x) (φ u)
          vφy : LE.le (φ u) (φ y)
          ux : LE.le u x
          ⊢ Membership.mem (Set.image φ (Inter.inter t (Set.Icc x y))) (φ u)
        -/
      · rw [← le_antisymm vφx (hφ us hx ux)]
        /-
          case h.e'_3.h.e'_6.a.intro.intro.intro.intro.inr
          α : Type u_1
          inst✝² : LinearOrder α
          E : Type u_2
          inst✝¹ : PseudoEMetricSpace E
          β : Type u_3
          inst✝ : LinearOrder β
          f : α → E
          t : Set β
          φ : β → α
          hφ : MonotoneOn φ t
          x y : β
          hx : Membership.mem t x
          hy : Membership.mem t y
          h : LE.le x y
          u : β
          us : Membership.mem t u
          vφx : LE.le (φ x) (φ u)
          vφy : LE.le (φ u) (φ y)
          ux : LE.le u x
          ⊢ Membership.mem (Set.image φ (Inter.inter t (Set.Icc x y))) (φ x)
        -/
        exact ⟨x, ⟨hx, ⟨le_rfl, h⟩⟩, rfl⟩
        /-
          🎉 no goals
        -/
      /-
        case h.e'_3.h.e'_6.a
        α : Type u_1
        inst✝² : LinearOrder α
        E : Type u_2
        inst✝¹ : PseudoEMetricSpace E
        β : Type u_3
        inst✝ : LinearOrder β
        f : α → E
        t : Set β
        φ : β → α
        hφ : MonotoneOn φ t
        x y : β
        hx : Membership.mem t x
        hy : Membership.mem t y
        h : LE.le x y
        ⊢ LE.le (Set.image φ (Inter.inter t (Set.Icc x y))) (Inter.inter (Set.image φ  …
      -/
    · rintro _ ⟨u, ⟨⟨hu, xu, uy⟩, rfl⟩⟩
      /-
        case h.e'_3.h.e'_6.a.intro.intro.intro.intro
        α : Type u_1
        inst✝² : LinearOrder α
        E : Type u_2
        inst✝¹ : PseudoEMetricSpace E
        β : Type u_3
        inst✝ : LinearOrder β
        f : α → E
        t : Set β
        φ : β → α
        hφ : MonotoneOn φ t
        x y : β
        hx : Membership.mem t x
        hy : Membership.mem t y
        h : LE.le x y
        u : β
        hu : Membership.mem t u
        xu : LE.le x u
        uy : LE.le u y
        ⊢ Membership.mem (Inter.inter (Set.image φ t) (Set.Icc (φ x) (φ y))) (φ u)
      -/
      exact ⟨⟨u, hu, rfl⟩, ⟨hφ hx hu xu, hφ hu hy uy⟩⟩
      /-
        🎉 no goals
      -/
    /-
      case inr
      α : Type u_1
      inst✝² : LinearOrder α
      E : Type u_2
      inst✝¹ : PseudoEMetricSpace E
      β : Type u_3
      inst✝ : LinearOrder β
      f : α → E
      t : Set β
      φ : β → α
      hφ : MonotoneOn φ t
      x y : β
      hx : Membership.mem t x
      hy : Membership.mem t y
      h : LE.le y x
      ⊢ Eq (eVariationOn (Function.comp f φ) (Inter.inter t (Set.Icc x y))) (eVariat …
    -/
  · rw [eVariationOn.subsingleton, eVariationOn.subsingleton]
    exacts [(Set.subsingleton_Icc_of_ge (hφ hy hx h)).anti Set.inter_subset_right,
      (Set.subsingleton_Icc_of_ge h).anti Set.inter_subset_right]


theorem comp_eq_of_antitoneOn (f : α → E) {t : Set β} (φ : β → α) (hφ : AntitoneOn φ t) :
    eVariationOn (f ∘ φ) t = eVariationOn f (φ '' t) := by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : AntitoneOn φ t
    ⊢ Eq (eVariationOn (Function.comp f φ) t) (eVariationOn f (Set.image φ t))
  -/
  apply le_antisymm (comp_le_of_antitoneOn f φ hφ (mapsTo_image φ t))
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : AntitoneOn φ t
    ⊢ LE.le (eVariationOn f (Set.image φ t)) (eVariationOn (Function.comp f φ) t)
  -/
  cases isEmpty_or_nonempty β
    /-
      case inl
      α : Type u_1
      inst✝² : LinearOrder α
      E : Type u_2
      inst✝¹ : PseudoEMetricSpace E
      β : Type u_3
      inst✝ : LinearOrder β
      f : α → E
      t : Set β
      φ : β → α
      hφ : AntitoneOn φ t
      h✝ : IsEmpty β
      ⊢ LE.le (eVariationOn f (Set.image φ t)) (eVariationOn (Function.comp f φ) t)
    -/
  · convert zero_le (_ : ℝ≥0∞)
    exact eVariationOn.subsingleton f <| (subsingleton_of_subsingleton.image _).anti
      (surjOn_image φ t)
  /-
    case inr
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : AntitoneOn φ t
    h✝ : Nonempty β
    ⊢ LE.le (eVariationOn f (Set.image φ t)) (eVariationOn (Function.comp f φ) t)
  -/
  let ψ := φ.invFunOn t
  /-
    case inr
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : AntitoneOn φ t
    h✝ : Nonempty β
    ψ : α → β := Function.invFunOn φ t
    ⊢ LE.le (eVariationOn f (Set.image φ t)) (eVariationOn (Function.comp f φ) t)
  -/
  have ψφs : EqOn (φ ∘ ψ) id (φ '' t) := (surjOn_image φ t).rightInvOn_invFunOn
  /-
    case inr
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : AntitoneOn φ t
    h✝ : Nonempty β
    ψ : α → β := Function.invFunOn φ t
    ψφs : Set.EqOn (Function.comp φ ψ) id (Set.image φ t)
    ⊢ LE.le (eVariationOn f (Set.image φ t)) (eVariationOn (Function.comp f φ) t)
  -/
  have ψts := (surjOn_image φ t).mapsTo_invFunOn
  /-
    case inr
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : AntitoneOn φ t
    h✝ : Nonempty β
    ψ : α → β := Function.invFunOn φ t
    ψφs : Set.EqOn (Function.comp φ ψ) id (Set.image φ t)
    ψts : Set.MapsTo (Function.invFunOn φ t) (Set.image φ t) t
    ⊢ LE.le (eVariationOn f (Set.image φ t)) (eVariationOn (Function.comp f φ) t)
  -/
  have hψ : AntitoneOn ψ (φ '' t) := Function.antitoneOn_of_rightInvOn_of_mapsTo hφ ψφs ψts
  /-
    case inr
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : AntitoneOn φ t
    h✝ : Nonempty β
    ψ : α → β := Function.invFunOn φ t
    ψφs : Set.EqOn (Function.comp φ ψ) id (Set.image φ t)
    ψts : Set.MapsTo (Function.invFunOn φ t) (Set.image φ t) t
    hψ : AntitoneOn ψ (Set.image φ t)
    ⊢ LE.le (eVariationOn f (Set.image φ t)) (eVariationOn (Function.comp f φ) t)
  -/
  change eVariationOn (f ∘ id) (φ '' t) ≤ eVariationOn (f ∘ φ) t
  /-
    case inr
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : AntitoneOn φ t
    h✝ : Nonempty β
    ψ : α → β := Function.invFunOn φ t
    ψφs : Set.EqOn (Function.comp φ ψ) id (Set.image φ t)
    ψts : Set.MapsTo (Function.invFunOn φ t) (Set.image φ t) t
    hψ : AntitoneOn ψ (Set.image φ t)
    ⊢ LE.le (eVariationOn (Function.comp f id) (Set.image φ t)) (eVariationOn (Fun …
  -/
  rw [← eq_of_eqOn (ψφs.comp_left : EqOn (f ∘ φ ∘ ψ) (f ∘ id) (φ '' t))]
  /-
    case inr
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : AntitoneOn φ t
    h✝ : Nonempty β
    ψ : α → β := Function.invFunOn φ t
    ψφs : Set.EqOn (Function.comp φ ψ) id (Set.image φ t)
    ψts : Set.MapsTo (Function.invFunOn φ t) (Set.image φ t) t
    hψ : AntitoneOn ψ (Set.image φ t)
    ⊢ LE.le (eVariationOn (Function.comp f (Function.comp φ ψ)) (Set.image φ t)) ( …
  -/
  exact comp_le_of_antitoneOn _ ψ hψ ψts
  /-
    🎉 no goals
  -/


theorem comp_ofDual (f : α → E) (s : Set α) :
    eVariationOn (f ∘ ofDual) (ofDual ⁻¹' s) = eVariationOn f s := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    ⊢ Eq (eVariationOn (Function.comp f ⇑OrderDual.ofDual) (Set.preimage (⇑OrderDu …
  -/
  convert comp_eq_of_antitoneOn f ofDual fun _ _ _ _ => id
  /-
    case h.e'_3.h.e'_6
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    ⊢ Eq s (Set.image (⇑OrderDual.ofDual) (Set.preimage (⇑OrderDual.ofDual) s))
  -/
  simp only [Equiv.image_preimage]
  /-
    🎉 no goals
  -/


theorem MonotoneOn.eVariationOn_le {f : α → ℝ} {s : Set α} (hf : MonotoneOn f s) {a b : α}
    (as : a ∈ s) (bs : b ∈ s) : eVariationOn f (s ∩ Icc a b) ≤ ENNReal.ofReal (f b - f a) := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    f : α → Real
    s : Set α
    hf : MonotoneOn f s
    a b : α
    as : Membership.mem s a
    bs : Membership.mem s b
    ⊢ LE.le (eVariationOn f (Inter.inter s (Set.Icc a b))) (ENNReal.ofReal (HSub.h …
  -/
  apply iSup_le _
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    f : α → Real
    s : Set α
    hf : MonotoneOn f s
    a b : α
    as : Membership.mem s a
    bs : Membership.mem s b
    ⊢ ∀ (i : Prod Nat (Subtype fun u => And (Monotone u) (∀ (i : Nat), Membership. …
  -/
  rintro ⟨n, ⟨u, hu, us⟩⟩
  calc
    (∑ i ∈ Finset.range n, edist (f (u (i + 1))) (f (u i))) =
        ∑ i ∈ Finset.range n, ENNReal.ofReal (f (u (i + 1)) - f (u i)) := by
      refine Finset.sum_congr rfl fun i hi => ?_
      simp only [Finset.mem_range] at hi
      rw [edist_dist, Real.dist_eq, abs_of_nonneg]
      exact sub_nonneg_of_le (hf (us i).1 (us (i + 1)).1 (hu (Nat.le_succ _)))
    _ = ENNReal.ofReal (∑ i ∈ Finset.range n, (f (u (i + 1)) - f (u i))) := by
      rw [ENNReal.ofReal_sum_of_nonneg]
      intro i _
      exact sub_nonneg_of_le (hf (us i).1 (us (i + 1)).1 (hu (Nat.le_succ _)))
    _ = ENNReal.ofReal (f (u n) - f (u 0)) := by rw [Finset.sum_range_sub fun i => f (u i)]
    _ ≤ ENNReal.ofReal (f b - f a) := by
      apply ENNReal.ofReal_le_ofReal
      exact sub_le_sub (hf (us n).1 bs (us n).2.2) (hf as (us 0).1 (us 0).2.1)


theorem MonotoneOn.locallyBoundedVariationOn {f : α → ℝ} {s : Set α} (hf : MonotoneOn f s) :
    LocallyBoundedVariationOn f s := fun _ _ as bs =>
  ((hf.eVariationOn_le as bs).trans_lt ENNReal.ofReal_lt_top).ne


/-- The **signed** variation of `f` on the interval `Icc a b` intersected with the set `s`,
squashed to a real (therefore only really meaningful if the variation is finite)
-/
noncomputable def variationOnFromTo (f : α → E) (s : Set α) (a b : α) : ℝ :=
  if a ≤ b then (eVariationOn f (s ∩ Icc a b)).toReal else -(eVariationOn f (s ∩ Icc b a)).toReal


protected theorem self (a : α) : variationOnFromTo f s a a = 0 := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    a : α
    ⊢ Eq (variationOnFromTo f s a a) 0
  -/
  dsimp only [variationOnFromTo]
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    a : α
    ⊢ Eq (ite (LE.le a a) (eVariationOn f (Inter.inter s (Set.Icc a a))).toReal (N …
  -/
  rw [if_pos le_rfl, Icc_self, eVariationOn.subsingleton, ENNReal.zero_toReal]
  /-
    case hs
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    a : α
    ⊢ (Inter.inter s (Singleton.singleton a)).Subsingleton
  -/
  exact fun x hx y hy => hx.2.trans hy.2.symm
  /-
    🎉 no goals
  -/


protected theorem nonneg_of_le {a b : α} (h : a ≤ b) : 0 ≤ variationOnFromTo f s a b := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    a b : α
    h : LE.le a b
    ⊢ LE.le 0 (variationOnFromTo f s a b)
  -/
  simp only [variationOnFromTo, if_pos h, ENNReal.toReal_nonneg]
  /-
    🎉 no goals
  -/


protected theorem eq_neg_swap (a b : α) :
    variationOnFromTo f s a b = -variationOnFromTo f s b a := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    a b : α
    ⊢ Eq (variationOnFromTo f s a b) (Neg.neg (variationOnFromTo f s b a))
  -/
  rcases lt_trichotomy a b with (ab | rfl | ba)
    /-
      case inl
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      a b : α
      ab : LT.lt a b
      ⊢ Eq (variationOnFromTo f s a b) (Neg.neg (variationOnFromTo f s b a))
    -/
  · simp only [variationOnFromTo, if_pos ab.le, if_neg ab.not_le, neg_neg]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      a : α
      ⊢ Eq (variationOnFromTo f s a a) (Neg.neg (variationOnFromTo f s a a))
    -/
  · simp only [variationOnFromTo.self, neg_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      a b : α
      ba : LT.lt b a
      ⊢ Eq (variationOnFromTo f s a b) (Neg.neg (variationOnFromTo f s b a))
    -/
  · simp only [variationOnFromTo, if_pos ba.le, if_neg ba.not_le, neg_neg]
    /-
      🎉 no goals
    -/


protected theorem nonpos_of_ge {a b : α} (h : b ≤ a) : variationOnFromTo f s a b ≤ 0 := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    a b : α
    h : LE.le b a
    ⊢ LE.le (variationOnFromTo f s a b) 0
  -/
  rw [variationOnFromTo.eq_neg_swap]
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    a b : α
    h : LE.le b a
    ⊢ LE.le (Neg.neg (variationOnFromTo f s b a)) 0
  -/
  exact neg_nonpos_of_nonneg (variationOnFromTo.nonneg_of_le f s h)
  /-
    🎉 no goals
  -/


protected theorem eq_of_le {a b : α} (h : a ≤ b) :
    variationOnFromTo f s a b = (eVariationOn f (s ∩ Icc a b)).toReal :=
  if_pos h


protected theorem eq_of_ge {a b : α} (h : b ≤ a) :
    variationOnFromTo f s a b = -(eVariationOn f (s ∩ Icc b a)).toReal := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    a b : α
    h : LE.le b a
    ⊢ Eq (variationOnFromTo f s a b) (Neg.neg (eVariationOn f (Inter.inter s (Set. …
  -/
  rw [variationOnFromTo.eq_neg_swap, neg_inj, variationOnFromTo.eq_of_le f s h]
  /-
    🎉 no goals
  -/


protected theorem add {f : α → E} {s : Set α} (hf : LocallyBoundedVariationOn f s) {a b c : α}
    (ha : a ∈ s) (hb : b ∈ s) (hc : c ∈ s) :
    variationOnFromTo f s a b + variationOnFromTo f s b c = variationOnFromTo f s a c := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    a b c : α
    ha : Membership.mem s a
    hb : Membership.mem s b
    hc : Membership.mem s c
    ⊢ Eq (HAdd.hAdd (variationOnFromTo f s a b) (variationOnFromTo f s b c)) (vari …
  -/
  symm
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    a b c : α
    ha : Membership.mem s a
    hb : Membership.mem s b
    hc : Membership.mem s c
    ⊢ Eq (variationOnFromTo f s a c) (HAdd.hAdd (variationOnFromTo f s a b) (varia …
  -/
  refine additive_of_isTotal ((· : α) ≤ ·) (variationOnFromTo f s) (· ∈ s) ?_ ?_ ha hb hc
    /-
      case refine_1
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      hf : LocallyBoundedVariationOn f s
      a b c : α
      ha : Membership.mem s a
      hb : Membership.mem s b
      hc : Membership.mem s c
      ⊢ ∀ {a b : α}, (fun x => Membership.mem s x) a → (fun x => Membership.mem s x) …
    -/
  · rintro x y _xs _ys
    simp only [variationOnFromTo.eq_neg_swap f s y x, Subtype.coe_mk, add_neg_cancel,
      forall_true_left]
    /-
      case refine_2
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      hf : LocallyBoundedVariationOn f s
      a b c : α
      ha : Membership.mem s a
      hb : Membership.mem s b
      hc : Membership.mem s c
      ⊢ ∀ {a b c : α}, (fun x1 x2 => LE.le x1 x2) a b → (fun x1 x2 => LE.le x1 x2) b …
    -/
  · rintro x y z xy yz xs ys zs
    rw [variationOnFromTo.eq_of_le f s xy, variationOnFromTo.eq_of_le f s yz,
      variationOnFromTo.eq_of_le f s (xy.trans yz),
      ← ENNReal.toReal_add (hf x y xs ys) (hf y z ys zs), eVariationOn.Icc_add_Icc f xy yz ys]


variable {f s} in
protected theorem edist_zero_of_eq_zero (hf : LocallyBoundedVariationOn f s)
    {a b : α} (ha : a ∈ s) (hb : b ∈ s) (h : variationOnFromTo f s a b = 0) :
    edist (f a) (f b) = 0 := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    a b : α
    ha : Membership.mem s a
    hb : Membership.mem s b
    h : Eq (variationOnFromTo f s a b) 0
    ⊢ Eq (EDist.edist (f a) (f b)) 0
  -/
  wlog h' : a ≤ b
    /-
      case inr
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      hf : LocallyBoundedVariationOn f s
      a b : α
      ha : Membership.mem s a
      hb : Membership.mem s b
      h : Eq (variationOnFromTo f s a b) 0
      this : ∀ {α : Type u_1} [inst : LinearOrder α] {E : Type u_2} [inst_1 : Pseudo …
      h' : Not (LE.le a b)
      ⊢ Eq (EDist.edist (f a) (f b)) 0
    -/
  · rw [edist_comm]
    /-
      case inr
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      hf : LocallyBoundedVariationOn f s
      a b : α
      ha : Membership.mem s a
      hb : Membership.mem s b
      h : Eq (variationOnFromTo f s a b) 0
      this : ∀ {α : Type u_1} [inst : LinearOrder α] {E : Type u_2} [inst_1 : Pseudo …
      h' : Not (LE.le a b)
      ⊢ Eq (EDist.edist (f b) (f a)) 0
    -/
    apply this hf hb ha _ (le_of_not_le h')
    /-
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      hf : LocallyBoundedVariationOn f s
      a b : α
      ha : Membership.mem s a
      hb : Membership.mem s b
      h : Eq (variationOnFromTo f s a b) 0
      this : ∀ {α : Type u_1} [inst : LinearOrder α] {E : Type u_2} [inst_1 : Pseudo …
      h' : Not (LE.le a b)
      ⊢ Eq (variationOnFromTo f s b a) 0
    -/
    rw [variationOnFromTo.eq_neg_swap, h, neg_zero]
    /-
      🎉 no goals
    -/
    /-
      α✝ : Type u_1
      inst✝³ : LinearOrder α✝
      E✝ : Type u_2
      inst✝² : PseudoEMetricSpace E✝
      f✝ : α✝ → E✝
      s✝ : Set α✝
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      hf : LocallyBoundedVariationOn f s
      a b : α
      ha : Membership.mem s a
      hb : Membership.mem s b
      h : Eq (variationOnFromTo f s a b) 0
      h' : LE.le a b
      ⊢ Eq (EDist.edist (f a) (f b)) 0
    -/
  · apply le_antisymm _ (zero_le _)
    rw [← ENNReal.ofReal_zero, ← h, variationOnFromTo.eq_of_le f s h',
      ENNReal.ofReal_toReal (hf a b ha hb)]
    /-
      α✝ : Type u_1
      inst✝³ : LinearOrder α✝
      E✝ : Type u_2
      inst✝² : PseudoEMetricSpace E✝
      f✝ : α✝ → E✝
      s✝ : Set α✝
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      hf : LocallyBoundedVariationOn f s
      a b : α
      ha : Membership.mem s a
      hb : Membership.mem s b
      h : Eq (variationOnFromTo f s a b) 0
      h' : LE.le a b
      ⊢ LE.le (EDist.edist (f a) (f b)) (eVariationOn f (Inter.inter s (Set.Icc a b)))
    -/
    apply eVariationOn.edist_le
    /-
      case hx
      α✝ : Type u_1
      inst✝³ : LinearOrder α✝
      E✝ : Type u_2
      inst✝² : PseudoEMetricSpace E✝
      f✝ : α✝ → E✝
      s✝ : Set α✝
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      hf : LocallyBoundedVariationOn f s
      a b : α
      ha : Membership.mem s a
      hb : Membership.mem s b
      h : Eq (variationOnFromTo f s a b) 0
      h' : LE.le a b
      ⊢ Membership.mem (Inter.inter s (Set.Icc a b)) a
    -/
    exacts [⟨ha, ⟨le_rfl, h'⟩⟩, ⟨hb, ⟨h', le_rfl⟩⟩]
    /-
      🎉 no goals
    -/


protected theorem eq_left_iff {f : α → E} {s : Set α} (hf : LocallyBoundedVariationOn f s)
    {a b c : α} (ha : a ∈ s) (hb : b ∈ s) (hc : c ∈ s) :
    variationOnFromTo f s a b = variationOnFromTo f s a c ↔ variationOnFromTo f s b c = 0 := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    a b c : α
    ha : Membership.mem s a
    hb : Membership.mem s b
    hc : Membership.mem s c
    ⊢ Iff (Eq (variationOnFromTo f s a b) (variationOnFromTo f s a c)) (Eq (variat …
  -/
  simp only [← variationOnFromTo.add hf ha hb hc, self_eq_add_right]
  /-
    🎉 no goals
  -/


protected theorem eq_zero_iff_of_le {f : α → E} {s : Set α} (hf : LocallyBoundedVariationOn f s)
    {a b : α} (ha : a ∈ s) (hb : b ∈ s) (ab : a ≤ b) :
    variationOnFromTo f s a b = 0 ↔
      ∀ ⦃x⦄ (_hx : x ∈ s ∩ Icc a b) ⦃y⦄ (_hy : y ∈ s ∩ Icc a b), edist (f x) (f y) = 0 := by
  rw [variationOnFromTo.eq_of_le _ _ ab, ENNReal.toReal_eq_zero_iff, or_iff_left (hf a b ha hb),
    eVariationOn.eq_zero_iff]


protected theorem eq_zero_iff_of_ge {f : α → E} {s : Set α} (hf : LocallyBoundedVariationOn f s)
    {a b : α} (ha : a ∈ s) (hb : b ∈ s) (ba : b ≤ a) :
    variationOnFromTo f s a b = 0 ↔
      ∀ ⦃x⦄ (_hx : x ∈ s ∩ Icc b a) ⦃y⦄ (_hy : y ∈ s ∩ Icc b a), edist (f x) (f y) = 0 := by
  rw [variationOnFromTo.eq_of_ge _ _ ba, neg_eq_zero, ENNReal.toReal_eq_zero_iff,
    or_iff_left (hf b a hb ha), eVariationOn.eq_zero_iff]


protected theorem eq_zero_iff {f : α → E} {s : Set α} (hf : LocallyBoundedVariationOn f s) {a b : α}
    (ha : a ∈ s) (hb : b ∈ s) :
    variationOnFromTo f s a b = 0 ↔
      ∀ ⦃x⦄ (_hx : x ∈ s ∩ uIcc a b) ⦃y⦄ (_hy : y ∈ s ∩ uIcc a b), edist (f x) (f y) = 0 := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    a b : α
    ha : Membership.mem s a
    hb : Membership.mem s b
    ⊢ Iff (Eq (variationOnFromTo f s a b) 0) (∀ ⦃x : α⦄, Membership.mem (Inter.int …
  -/
  rcases le_total a b with (ab | ba)
    /-
      case inl
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      hf : LocallyBoundedVariationOn f s
      a b : α
      ha : Membership.mem s a
      hb : Membership.mem s b
      ab : LE.le a b
      ⊢ Iff (Eq (variationOnFromTo f s a b) 0) (∀ ⦃x : α⦄, Membership.mem (Inter.int …
    -/
  · rw [uIcc_of_le ab]
    /-
      case inl
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      hf : LocallyBoundedVariationOn f s
      a b : α
      ha : Membership.mem s a
      hb : Membership.mem s b
      ab : LE.le a b
      ⊢ Iff (Eq (variationOnFromTo f s a b) 0) (∀ ⦃x : α⦄, Membership.mem (Inter.int …
    -/
    exact variationOnFromTo.eq_zero_iff_of_le hf ha hb ab
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      hf : LocallyBoundedVariationOn f s
      a b : α
      ha : Membership.mem s a
      hb : Membership.mem s b
      ba : LE.le b a
      ⊢ Iff (Eq (variationOnFromTo f s a b) 0) (∀ ⦃x : α⦄, Membership.mem (Inter.int …
    -/
  · rw [uIcc_of_ge ba]
    /-
      case inr
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      hf : LocallyBoundedVariationOn f s
      a b : α
      ha : Membership.mem s a
      hb : Membership.mem s b
      ba : LE.le b a
      ⊢ Iff (Eq (variationOnFromTo f s a b) 0) (∀ ⦃x : α⦄, Membership.mem (Inter.int …
    -/
    exact variationOnFromTo.eq_zero_iff_of_ge hf ha hb ba
    /-
      🎉 no goals
    -/


protected theorem monotoneOn (hf : LocallyBoundedVariationOn f s) {a : α} (as : a ∈ s) :
    MonotoneOn (variationOnFromTo f s a) s := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    a : α
    as : Membership.mem s a
    ⊢ MonotoneOn (variationOnFromTo f s a) s
  -/
  rintro b bs c cs bc
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    a : α
    as : Membership.mem s a
    b : α
    bs : Membership.mem s b
    c : α
    cs : Membership.mem s c
    bc : LE.le b c
    ⊢ LE.le (variationOnFromTo f s a b) (variationOnFromTo f s a c)
  -/
  rw [← variationOnFromTo.add hf as bs cs]
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    a : α
    as : Membership.mem s a
    b : α
    bs : Membership.mem s b
    c : α
    cs : Membership.mem s c
    bc : LE.le b c
    ⊢ LE.le (variationOnFromTo f s a b) (HAdd.hAdd (variationOnFromTo f s a b) (va …
  -/
  exact le_add_of_nonneg_right (variationOnFromTo.nonneg_of_le f s bc)
  /-
    🎉 no goals
  -/


protected theorem antitoneOn (hf : LocallyBoundedVariationOn f s) {b : α} (bs : b ∈ s) :
    AntitoneOn (fun a => variationOnFromTo f s a b) s := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    b : α
    bs : Membership.mem s b
    ⊢ AntitoneOn (fun a => variationOnFromTo f s a b) s
  -/
  rintro a as c cs ac
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    b : α
    bs : Membership.mem s b
    a : α
    as : Membership.mem s a
    c : α
    cs : Membership.mem s c
    ac : LE.le a c
    ⊢ LE.le ((fun a => variationOnFromTo f s a b) c) ((fun a => variationOnFromTo  …
  -/
  dsimp only
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    b : α
    bs : Membership.mem s b
    a : α
    as : Membership.mem s a
    c : α
    cs : Membership.mem s c
    ac : LE.le a c
    ⊢ LE.le (variationOnFromTo f s c b) (variationOnFromTo f s a b)
  -/
  rw [← variationOnFromTo.add hf as cs bs]
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    b : α
    bs : Membership.mem s b
    a : α
    as : Membership.mem s a
    c : α
    cs : Membership.mem s c
    ac : LE.le a c
    ⊢ LE.le (variationOnFromTo f s c b) (HAdd.hAdd (variationOnFromTo f s a c) (va …
  -/
  exact le_add_of_nonneg_left (variationOnFromTo.nonneg_of_le f s ac)
  /-
    🎉 no goals
  -/


protected theorem sub_self_monotoneOn {f : α → ℝ} {s : Set α} (hf : LocallyBoundedVariationOn f s)
    {a : α} (as : a ∈ s) : MonotoneOn (variationOnFromTo f s a - f) s := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    f : α → Real
    s : Set α
    hf : LocallyBoundedVariationOn f s
    a : α
    as : Membership.mem s a
    ⊢ MonotoneOn (HSub.hSub (variationOnFromTo f s a) f) s
  -/
  rintro b bs c cs bc
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    f : α → Real
    s : Set α
    hf : LocallyBoundedVariationOn f s
    a : α
    as : Membership.mem s a
    b : α
    bs : Membership.mem s b
    c : α
    cs : Membership.mem s c
    bc : LE.le b c
    ⊢ LE.le (HSub.hSub (variationOnFromTo f s a) f b) (HSub.hSub (variationOnFromT …
  -/
  rw [Pi.sub_apply, Pi.sub_apply, le_sub_iff_add_le, add_comm_sub, ← le_sub_iff_add_le']
  calc
    f c - f b ≤ |f c - f b| := le_abs_self _
    _ = dist (f b) (f c) := by rw [dist_comm, Real.dist_eq]
    _ ≤ variationOnFromTo f s b c := by
      rw [variationOnFromTo.eq_of_le f s bc, dist_edist]
      apply ENNReal.toReal_mono (hf b c bs cs)
      apply eVariationOn.edist_le f
      exacts [⟨bs, le_rfl, bc⟩, ⟨cs, bc, le_rfl⟩]
    _ = variationOnFromTo f s a c - variationOnFromTo f s a b := by
      rw [← variationOnFromTo.add hf as bs cs, add_sub_cancel_left]


protected theorem comp_eq_of_monotoneOn {β : Type*} [LinearOrder β] (f : α → E) {t : Set β}
    (φ : β → α) (hφ : MonotoneOn φ t) {x y : β} (hx : x ∈ t) (hy : y ∈ t) :
    variationOnFromTo (f ∘ φ) t x y = variationOnFromTo f (φ '' t) (φ x) (φ y) := by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    β : Type u_3
    inst✝ : LinearOrder β
    f : α → E
    t : Set β
    φ : β → α
    hφ : MonotoneOn φ t
    x y : β
    hx : Membership.mem t x
    hy : Membership.mem t y
    ⊢ Eq (variationOnFromTo (Function.comp f φ) t x y) (variationOnFromTo f (Set.i …
  -/
  rcases le_total x y with (h | h)
  · rw [variationOnFromTo.eq_of_le _ _ h, variationOnFromTo.eq_of_le _ _ (hφ hx hy h),
      eVariationOn.comp_inter_Icc_eq_of_monotoneOn f φ hφ hx hy]
  · rw [variationOnFromTo.eq_of_ge _ _ h, variationOnFromTo.eq_of_ge _ _ (hφ hy hx h),
      eVariationOn.comp_inter_Icc_eq_of_monotoneOn f φ hφ hy hx]


/-- If a real valued function has bounded variation on a set, then it is a difference of monotone
functions there. -/
theorem LocallyBoundedVariationOn.exists_monotoneOn_sub_monotoneOn {f : α → ℝ} {s : Set α}
    (h : LocallyBoundedVariationOn f s) :
    ∃ p q : α → ℝ, MonotoneOn p s ∧ MonotoneOn q s ∧ f = p - q := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    f : α → Real
    s : Set α
    h : LocallyBoundedVariationOn f s
    ⊢ Exists fun p => Exists fun q => And (MonotoneOn p s) (And (MonotoneOn q s) ( …
  -/
  rcases eq_empty_or_nonempty s with (rfl | ⟨c, cs⟩)
  · exact ⟨f, 0, subsingleton_empty.monotoneOn _, subsingleton_empty.monotoneOn _,
      (sub_zero f).symm⟩
  · exact ⟨_, _, variationOnFromTo.monotoneOn h cs, variationOnFromTo.sub_self_monotoneOn h cs,
      (sub_sub_cancel _ _).symm⟩


theorem LipschitzOnWith.comp_eVariationOn_le {f : E → F} {C : ℝ≥0} {t : Set E}
    (h : LipschitzOnWith C f t) {g : α → E} {s : Set α} (hg : MapsTo g s t) :
    eVariationOn (f ∘ g) s ≤ C * eVariationOn g s := by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    F : Type u_3
    inst✝ : PseudoEMetricSpace F
    f : E → F
    C : NNReal
    t : Set E
    h : LipschitzOnWith C f t
    g : α → E
    s : Set α
    hg : Set.MapsTo g s t
    ⊢ LE.le (eVariationOn (Function.comp f g) s) (HMul.hMul (↑C) (eVariationOn g s))
  -/
  apply iSup_le _
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    E : Type u_2
    inst✝¹ : PseudoEMetricSpace E
    F : Type u_3
    inst✝ : PseudoEMetricSpace F
    f : E → F
    C : NNReal
    t : Set E
    h : LipschitzOnWith C f t
    g : α → E
    s : Set α
    hg : Set.MapsTo g s t
    ⊢ ∀ (i : Prod Nat (Subtype fun u => And (Monotone u) (∀ (i : Nat), Membership. …
  -/
  rintro ⟨n, ⟨u, hu, us⟩⟩
  calc
    (∑ i ∈ Finset.range n, edist (f (g (u (i + 1)))) (f (g (u i)))) ≤
        ∑ i ∈ Finset.range n, C * edist (g (u (i + 1))) (g (u i)) :=
      Finset.sum_le_sum fun i _ => h (hg (us _)) (hg (us _))
    _ = C * ∑ i ∈ Finset.range n, edist (g (u (i + 1))) (g (u i)) := by rw [Finset.mul_sum]
    _ ≤ C * eVariationOn g s := mul_le_mul_left' (eVariationOn.sum_le _ _ hu us) _


theorem LipschitzOnWith.comp_boundedVariationOn {f : E → F} {C : ℝ≥0} {t : Set E}
    (hf : LipschitzOnWith C f t) {g : α → E} {s : Set α} (hg : MapsTo g s t)
    (h : BoundedVariationOn g s) : BoundedVariationOn (f ∘ g) s :=
  ne_top_of_le_ne_top (ENNReal.mul_ne_top ENNReal.coe_ne_top h) (hf.comp_eVariationOn_le hg)


theorem LipschitzOnWith.comp_locallyBoundedVariationOn {f : E → F} {C : ℝ≥0} {t : Set E}
    (hf : LipschitzOnWith C f t) {g : α → E} {s : Set α} (hg : MapsTo g s t)
    (h : LocallyBoundedVariationOn g s) : LocallyBoundedVariationOn (f ∘ g) s :=
  fun x y xs ys =>
  hf.comp_boundedVariationOn (hg.mono_left inter_subset_left) (h x y xs ys)


theorem LipschitzWith.comp_boundedVariationOn {f : E → F} {C : ℝ≥0} (hf : LipschitzWith C f)
    {g : α → E} {s : Set α} (h : BoundedVariationOn g s) : BoundedVariationOn (f ∘ g) s :=
  hf.lipschitzOnWith.comp_boundedVariationOn (mapsTo_univ _ _) h


theorem LipschitzWith.comp_locallyBoundedVariationOn {f : E → F} {C : ℝ≥0}
    (hf : LipschitzWith C f) {g : α → E} {s : Set α} (h : LocallyBoundedVariationOn g s) :
    LocallyBoundedVariationOn (f ∘ g) s :=
  hf.lipschitzOnWith.comp_locallyBoundedVariationOn (mapsTo_univ _ _) h


theorem LipschitzOnWith.locallyBoundedVariationOn {f : ℝ → E} {C : ℝ≥0} {s : Set ℝ}
    (hf : LipschitzOnWith C f s) : LocallyBoundedVariationOn f s :=
  hf.comp_locallyBoundedVariationOn (mapsTo_id _)
    (@monotoneOn_id ℝ _ s).locallyBoundedVariationOn


theorem LipschitzWith.locallyBoundedVariationOn {f : ℝ → E} {C : ℝ≥0} (hf : LipschitzWith C f)
    (s : Set ℝ) : LocallyBoundedVariationOn f s :=
  hf.lipschitzOnWith.locallyBoundedVariationOn



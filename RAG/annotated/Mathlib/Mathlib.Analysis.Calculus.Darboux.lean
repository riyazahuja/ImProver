/-- **Darboux's theorem**: if `a ≤ b` and `f' a < m < f' b`, then `f' c = m` for some
`c ∈ (a, b)`. -/
theorem exists_hasDerivWithinAt_eq_of_gt_of_lt (hab : a ≤ b)
    (hf : ∀ x ∈ Icc a b, HasDerivWithinAt f (f' x) (Icc a b) x) {m : ℝ} (hma : f' a < m)
    (hmb : m < f' b) : m ∈ f' '' Ioo a b := by
  /-
    a b : Real
    f f' : Real → Real
    hab : LE.le a b
    hf : ∀ (x : Real), Membership.mem (Set.Icc a b) x → HasDerivWithinAt f (f' x)  …
    m : Real
    hma : LT.lt (f' a) m
    hmb : LT.lt m (f' b)
    ⊢ Membership.mem (Set.image f' (Set.Ioo a b)) m
  -/
  rcases hab.eq_or_lt with (rfl | hab')
    /-
      case inl
      a : Real
      f f' : Real → Real
      m : Real
      hma : LT.lt (f' a) m
      hab : LE.le a a
      hf : ∀ (x : Real), Membership.mem (Set.Icc a a) x → HasDerivWithinAt f (f' x)  …
      hmb : LT.lt m (f' a)
      ⊢ Membership.mem (Set.image f' (Set.Ioo a a)) m
    -/
  · exact (lt_asymm hma hmb).elim
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Real
    f f' : Real → Real
    hab : LE.le a b
    hf : ∀ (x : Real), Membership.mem (Set.Icc a b) x → HasDerivWithinAt f (f' x)  …
    m : Real
    hma : LT.lt (f' a) m
    hmb : LT.lt m (f' b)
    hab' : LT.lt a b
    ⊢ Membership.mem (Set.image f' (Set.Ioo a b)) m
  -/
  set g : ℝ → ℝ := fun x => f x - m * x
  have hg : ∀ x ∈ Icc a b, HasDerivWithinAt g (f' x - m) (Icc a b) x := by
    intro x hx
    simpa using (hf x hx).sub ((hasDerivWithinAt_id x _).const_mul m)
  obtain ⟨c, cmem, hc⟩ : ∃ c ∈ Icc a b, IsMinOn g (Icc a b) c :=
    isCompact_Icc.exists_isMinOn (nonempty_Icc.2 <| hab) fun x hx => (hg x hx).continuousWithinAt
  have cmem' : c ∈ Ioo a b := by
    rcases cmem.1.eq_or_lt with (rfl | hac)
    -- Show that `c` can't be equal to `a`
    · refine absurd (sub_nonneg.1 <| nonneg_of_mul_nonneg_right ?_ (sub_pos.2 hab'))
        (not_le_of_lt hma)
      have : b - a ∈ posTangentConeAt (Icc a b) a :=
        sub_mem_posTangentConeAt_of_segment_subset (segment_eq_Icc hab ▸ Subset.rfl)
      simpa only [ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.one_apply]
        using hc.localize.hasFDerivWithinAt_nonneg (hg a (left_mem_Icc.2 hab)) this
    rcases cmem.2.eq_or_gt with (rfl | hcb)
    -- Show that `c` can't be equal to `b`
    · refine absurd (sub_nonpos.1 <| nonpos_of_mul_nonneg_right ?_ (sub_lt_zero.2 hab'))
        (not_le_of_lt hmb)
      have : a - b ∈ posTangentConeAt (Icc a b) b :=
        sub_mem_posTangentConeAt_of_segment_subset (by rw [segment_symm, segment_eq_Icc hab])
      simpa only [ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.one_apply]
        using hc.localize.hasFDerivWithinAt_nonneg (hg b (right_mem_Icc.2 hab)) this
    exact ⟨hac, hcb⟩
  /-
    case inr.intro.intro
    a b : Real
    f f' : Real → Real
    hab : LE.le a b
    hf : ∀ (x : Real), Membership.mem (Set.Icc a b) x → HasDerivWithinAt f (f' x)  …
    m : Real
    hma : LT.lt (f' a) m
    hmb : LT.lt m (f' b)
    hab' : LT.lt a b
    g : Real → Real := fun x => HSub.hSub (f x) (HMul.hMul m x)
    hg : ∀ (x : Real), Membership.mem (Set.Icc a b) x → HasDerivWithinAt g (HSub.h …
    c : Real
    cmem : Membership.mem (Set.Icc a b) c
    hc : IsMinOn g (Set.Icc a b) c
    cmem' : Membership.mem (Set.Ioo a b) c
    ⊢ Membership.mem (Set.image f' (Set.Ioo a b)) m
  -/
  use c, cmem'
  /-
    case right
    a b : Real
    f f' : Real → Real
    hab : LE.le a b
    hf : ∀ (x : Real), Membership.mem (Set.Icc a b) x → HasDerivWithinAt f (f' x)  …
    m : Real
    hma : LT.lt (f' a) m
    hmb : LT.lt m (f' b)
    hab' : LT.lt a b
    g : Real → Real := fun x => HSub.hSub (f x) (HMul.hMul m x)
    hg : ∀ (x : Real), Membership.mem (Set.Icc a b) x → HasDerivWithinAt g (HSub.h …
    c : Real
    cmem : Membership.mem (Set.Icc a b) c
    hc : IsMinOn g (Set.Icc a b) c
    cmem' : Membership.mem (Set.Ioo a b) c
    ⊢ Eq (f' c) m
  -/
  rw [← sub_eq_zero]
  /-
    case right
    a b : Real
    f f' : Real → Real
    hab : LE.le a b
    hf : ∀ (x : Real), Membership.mem (Set.Icc a b) x → HasDerivWithinAt f (f' x)  …
    m : Real
    hma : LT.lt (f' a) m
    hmb : LT.lt m (f' b)
    hab' : LT.lt a b
    g : Real → Real := fun x => HSub.hSub (f x) (HMul.hMul m x)
    hg : ∀ (x : Real), Membership.mem (Set.Icc a b) x → HasDerivWithinAt g (HSub.h …
    c : Real
    cmem : Membership.mem (Set.Icc a b) c
    hc : IsMinOn g (Set.Icc a b) c
    cmem' : Membership.mem (Set.Ioo a b) c
    ⊢ Eq (HSub.hSub (f' c) m) 0
  -/
  have : Icc a b ∈ 𝓝 c := by rwa [← mem_interior_iff_mem_nhds, interior_Icc]
  /-
    case right
    a b : Real
    f f' : Real → Real
    hab : LE.le a b
    hf : ∀ (x : Real), Membership.mem (Set.Icc a b) x → HasDerivWithinAt f (f' x)  …
    m : Real
    hma : LT.lt (f' a) m
    hmb : LT.lt m (f' b)
    hab' : LT.lt a b
    g : Real → Real := fun x => HSub.hSub (f x) (HMul.hMul m x)
    hg : ∀ (x : Real), Membership.mem (Set.Icc a b) x → HasDerivWithinAt g (HSub.h …
    c : Real
    cmem : Membership.mem (Set.Icc a b) c
    hc : IsMinOn g (Set.Icc a b) c
    cmem' : Membership.mem (Set.Ioo a b) c
    this : Membership.mem (nhds c) (Set.Icc a b)
    ⊢ Eq (HSub.hSub (f' c) m) 0
  -/
  exact (hc.isLocalMin this).hasDerivAt_eq_zero ((hg c cmem).hasDerivAt this)
  /-
    🎉 no goals
  -/


/-- **Darboux's theorem**: if `a ≤ b` and `f' b < m < f' a`, then `f' c = m` for some `c ∈ (a, b)`.
-/
theorem exists_hasDerivWithinAt_eq_of_lt_of_gt (hab : a ≤ b)
    (hf : ∀ x ∈ Icc a b, HasDerivWithinAt f (f' x) (Icc a b) x) {m : ℝ} (hma : m < f' a)
    (hmb : f' b < m) : m ∈ f' '' Ioo a b :=
  let ⟨c, cmem, hc⟩ :=
    exists_hasDerivWithinAt_eq_of_gt_of_lt hab (fun x hx => (hf x hx).neg) (neg_lt_neg hma)
      (neg_lt_neg hmb)
  ⟨c, cmem, neg_injective hc⟩


/-- **Darboux's theorem**: the image of a `Set.OrdConnected` set under `f'` is a `Set.OrdConnected`
set, `HasDerivWithinAt` version. -/
theorem Set.OrdConnected.image_hasDerivWithinAt {s : Set ℝ} (hs : OrdConnected s)
    (hf : ∀ x ∈ s, HasDerivWithinAt f (f' x) s x) : OrdConnected (f' '' s) := by
  /-
    f f' : Real → Real
    s : Set Real
    hs : s.OrdConnected
    hf : ∀ (x : Real), Membership.mem s x → HasDerivWithinAt f (f' x) s x
    ⊢ (Set.image f' s).OrdConnected
  -/
  apply ordConnected_of_Ioo
  /-
    case hs
    f f' : Real → Real
    s : Set Real
    hs : s.OrdConnected
    hf : ∀ (x : Real), Membership.mem s x → HasDerivWithinAt f (f' x) s x
    ⊢ ∀ (x : Real), Membership.mem (Set.image f' s) x → ∀ (y : Real), Membership.m …
  -/
  rintro _ ⟨a, ha, rfl⟩ _ ⟨b, hb, rfl⟩ - m ⟨hma, hmb⟩
  /-
    case hs.intro.intro.intro.intro.intro
    f f' : Real → Real
    s : Set Real
    hs : s.OrdConnected
    hf : ∀ (x : Real), Membership.mem s x → HasDerivWithinAt f (f' x) s x
    a : Real
    ha : Membership.mem s a
    b : Real
    hb : Membership.mem s b
    m : Real
    hma : LT.lt (f' a) m
    hmb : LT.lt m (f' b)
    ⊢ Membership.mem (Set.image f' s) m
  -/
  rcases le_total a b with hab | hab
    /-
      case hs.intro.intro.intro.intro.intro.inl
      f f' : Real → Real
      s : Set Real
      hs : s.OrdConnected
      hf : ∀ (x : Real), Membership.mem s x → HasDerivWithinAt f (f' x) s x
      a : Real
      ha : Membership.mem s a
      b : Real
      hb : Membership.mem s b
      m : Real
      hma : LT.lt (f' a) m
      hmb : LT.lt m (f' b)
      hab : LE.le a b
      ⊢ Membership.mem (Set.image f' s) m
    -/
  · have : Icc a b ⊆ s := hs.out ha hb
    rcases exists_hasDerivWithinAt_eq_of_gt_of_lt hab (fun x hx => (hf x <| this hx).mono this) hma
        hmb with
      ⟨c, cmem, hc⟩
    /-
      case hs.intro.intro.intro.intro.intro.inl.intro.intro
      f f' : Real → Real
      s : Set Real
      hs : s.OrdConnected
      hf : ∀ (x : Real), Membership.mem s x → HasDerivWithinAt f (f' x) s x
      a : Real
      ha : Membership.mem s a
      b : Real
      hb : Membership.mem s b
      m : Real
      hma : LT.lt (f' a) m
      hmb : LT.lt m (f' b)
      hab : LE.le a b
      this : HasSubset.Subset (Set.Icc a b) s
      c : Real
      cmem : Membership.mem (Set.Ioo a b) c
      hc : Eq (f' c) m
      ⊢ Membership.mem (Set.image f' s) m
    -/
    exact ⟨c, this <| Ioo_subset_Icc_self cmem, hc⟩
    /-
      🎉 no goals
    -/
    /-
      case hs.intro.intro.intro.intro.intro.inr
      f f' : Real → Real
      s : Set Real
      hs : s.OrdConnected
      hf : ∀ (x : Real), Membership.mem s x → HasDerivWithinAt f (f' x) s x
      a : Real
      ha : Membership.mem s a
      b : Real
      hb : Membership.mem s b
      m : Real
      hma : LT.lt (f' a) m
      hmb : LT.lt m (f' b)
      hab : LE.le b a
      ⊢ Membership.mem (Set.image f' s) m
    -/
  · have : Icc b a ⊆ s := hs.out hb ha
    rcases exists_hasDerivWithinAt_eq_of_lt_of_gt hab (fun x hx => (hf x <| this hx).mono this) hmb
        hma with
      ⟨c, cmem, hc⟩
    /-
      case hs.intro.intro.intro.intro.intro.inr.intro.intro
      f f' : Real → Real
      s : Set Real
      hs : s.OrdConnected
      hf : ∀ (x : Real), Membership.mem s x → HasDerivWithinAt f (f' x) s x
      a : Real
      ha : Membership.mem s a
      b : Real
      hb : Membership.mem s b
      m : Real
      hma : LT.lt (f' a) m
      hmb : LT.lt m (f' b)
      hab : LE.le b a
      this : HasSubset.Subset (Set.Icc b a) s
      c : Real
      cmem : Membership.mem (Set.Ioo b a) c
      hc : Eq (f' c) m
      ⊢ Membership.mem (Set.image f' s) m
    -/
    exact ⟨c, this <| Ioo_subset_Icc_self cmem, hc⟩
    /-
      🎉 no goals
    -/


/-- **Darboux's theorem**: the image of a `Set.OrdConnected` set under `f'` is a `Set.OrdConnected`
set, `derivWithin` version. -/
theorem Set.OrdConnected.image_derivWithin {s : Set ℝ} (hs : OrdConnected s)
    (hf : DifferentiableOn ℝ f s) : OrdConnected (derivWithin f s '' s) :=
  hs.image_hasDerivWithinAt fun x hx => (hf x hx).hasDerivWithinAt


/-- **Darboux's theorem**: the image of a `Set.OrdConnected` set under `f'` is a `Set.OrdConnected`
set, `deriv` version. -/
theorem Set.OrdConnected.image_deriv {s : Set ℝ} (hs : OrdConnected s)
    (hf : ∀ x ∈ s, DifferentiableAt ℝ f x) : OrdConnected (deriv f '' s) :=
  hs.image_hasDerivWithinAt fun x hx => (hf x hx).hasDerivAt.hasDerivWithinAt


/-- **Darboux's theorem**: the image of a convex set under `f'` is a convex set,
`HasDerivWithinAt` version. -/
theorem Convex.image_hasDerivWithinAt {s : Set ℝ} (hs : Convex ℝ s)
    (hf : ∀ x ∈ s, HasDerivWithinAt f (f' x) s x) : Convex ℝ (f' '' s) :=
  (hs.ordConnected.image_hasDerivWithinAt hf).convex


/-- **Darboux's theorem**: the image of a convex set under `f'` is a convex set,
`derivWithin` version. -/
theorem Convex.image_derivWithin {s : Set ℝ} (hs : Convex ℝ s) (hf : DifferentiableOn ℝ f s) :
    Convex ℝ (derivWithin f s '' s) :=
  (hs.ordConnected.image_derivWithin hf).convex


/-- **Darboux's theorem**: the image of a convex set under `f'` is a convex set,
`deriv` version. -/
theorem Convex.image_deriv {s : Set ℝ} (hs : Convex ℝ s) (hf : ∀ x ∈ s, DifferentiableAt ℝ f x) :
    Convex ℝ (deriv f '' s) :=
  (hs.ordConnected.image_deriv hf).convex


/-- **Darboux's theorem**: if `a ≤ b` and `f' a ≤ m ≤ f' b`, then `f' c = m` for some
`c ∈ [a, b]`. -/
theorem exists_hasDerivWithinAt_eq_of_ge_of_le (hab : a ≤ b)
    (hf : ∀ x ∈ Icc a b, HasDerivWithinAt f (f' x) (Icc a b) x) {m : ℝ} (hma : f' a ≤ m)
    (hmb : m ≤ f' b) : m ∈ f' '' Icc a b :=
  (ordConnected_Icc.image_hasDerivWithinAt hf).out (mem_image_of_mem _ (left_mem_Icc.2 hab))
    (mem_image_of_mem _ (right_mem_Icc.2 hab)) ⟨hma, hmb⟩


/-- **Darboux's theorem**: if `a ≤ b` and `f' b ≤ m ≤ f' a`, then `f' c = m` for some
`c ∈ [a, b]`. -/
theorem exists_hasDerivWithinAt_eq_of_le_of_ge (hab : a ≤ b)
    (hf : ∀ x ∈ Icc a b, HasDerivWithinAt f (f' x) (Icc a b) x) {m : ℝ} (hma : f' a ≤ m)
    (hmb : m ≤ f' b) : m ∈ f' '' Icc a b :=
  (ordConnected_Icc.image_hasDerivWithinAt hf).out (mem_image_of_mem _ (left_mem_Icc.2 hab))
    (mem_image_of_mem _ (right_mem_Icc.2 hab)) ⟨hma, hmb⟩


/-- If the derivative of a function is never equal to `m`, then either
it is always greater than `m`, or it is always less than `m`. -/
theorem hasDerivWithinAt_forall_lt_or_forall_gt_of_forall_ne {s : Set ℝ} (hs : Convex ℝ s)
    (hf : ∀ x ∈ s, HasDerivWithinAt f (f' x) s x) {m : ℝ} (hf' : ∀ x ∈ s, f' x ≠ m) :
    (∀ x ∈ s, f' x < m) ∨ ∀ x ∈ s, m < f' x := by
  /-
    f f' : Real → Real
    s : Set Real
    hs : Convex Real s
    hf : ∀ (x : Real), Membership.mem s x → HasDerivWithinAt f (f' x) s x
    m : Real
    hf' : ∀ (x : Real), Membership.mem s x → Ne (f' x) m
    ⊢ Or (∀ (x : Real), Membership.mem s x → LT.lt (f' x) m) (∀ (x : Real), Member …
  -/
  contrapose! hf'
  /-
    f f' : Real → Real
    s : Set Real
    hs : Convex Real s
    hf : ∀ (x : Real), Membership.mem s x → HasDerivWithinAt f (f' x) s x
    m : Real
    hf' : And (Exists fun x => And (Membership.mem s x) (LE.le m (f' x))) (Exists  …
    ⊢ Exists fun x => And (Membership.mem s x) (Eq (f' x) m)
  -/
  rcases hf' with ⟨⟨b, hb, hmb⟩, ⟨a, ha, hma⟩⟩
  exact (hs.ordConnected.image_hasDerivWithinAt hf).out (mem_image_of_mem f' ha)
    (mem_image_of_mem f' hb) ⟨hma, hmb⟩


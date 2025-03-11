/-- If `f` is a function strictly monotone on a right neighborhood of `a` and the
image of this neighborhood under `f` meets every interval `(f a, b]`, `b > f a`, then `f` is
continuous at `a` from the right.

The assumption `hfs : ∀ b > f a, ∃ c ∈ s, f c ∈ Ioc (f a) b` is required because otherwise the
function `f : ℝ → ℝ` given by `f x = if x ≤ 0 then x else x + 1` would be a counter-example at
`a = 0`. -/
theorem StrictMonoOn.continuousWithinAt_right_of_exists_between {f : α → β} {s : Set α} {a : α}
    (h_mono : StrictMonoOn f s) (hs : s ∈ 𝓝[≥] a) (hfs : ∀ b > f a, ∃ c ∈ s, f c ∈ Ioc (f a) b) :
    ContinuousWithinAt f (Ici a) a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    inst✝² : LinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    s : Set α
    a : α
    h_mono : StrictMonoOn f s
    hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
    hfs : ∀ (b : β), GT.gt b (f a) → Exists fun c => And (Membership.mem s c) (Mem …
    ⊢ ContinuousWithinAt f (Set.Ici a) a
  -/
  have ha : a ∈ Ici a := left_mem_Ici
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    inst✝² : LinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    s : Set α
    a : α
    h_mono : StrictMonoOn f s
    hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
    hfs : ∀ (b : β), GT.gt b (f a) → Exists fun c => And (Membership.mem s c) (Mem …
    ha : Membership.mem (Set.Ici a) a
    ⊢ ContinuousWithinAt f (Set.Ici a) a
  -/
  have has : a ∈ s := mem_of_mem_nhdsWithin ha hs
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    inst✝² : LinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    s : Set α
    a : α
    h_mono : StrictMonoOn f s
    hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
    hfs : ∀ (b : β), GT.gt b (f a) → Exists fun c => And (Membership.mem s c) (Mem …
    ha : Membership.mem (Set.Ici a) a
    has : Membership.mem s a
    ⊢ ContinuousWithinAt f (Set.Ici a) a
  -/
  refine tendsto_order.2 ⟨fun b hb => ?_, fun b hb => ?_⟩
  · filter_upwards [hs, @self_mem_nhdsWithin _ _ a (Ici a)] with _ hxs hxa using hb.trans_le
      ((h_mono.le_iff_le has hxs).2 hxa)
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : LinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      s : Set α
      a : α
      h_mono : StrictMonoOn f s
      hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
      hfs : ∀ (b : β), GT.gt b (f a) → Exists fun c => And (Membership.mem s c) (Mem …
      ha : Membership.mem (Set.Ici a) a
      has : Membership.mem s a
      b : β
      hb : GT.gt b (f a)
      ⊢ Filter.Eventually (fun b_1 => LT.lt (f b_1) b) (nhdsWithin a (Set.Ici a))
    -/
  · rcases hfs b hb with ⟨c, hcs, hac, hcb⟩
    /-
      case refine_2.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : LinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      s : Set α
      a : α
      h_mono : StrictMonoOn f s
      hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
      hfs : ∀ (b : β), GT.gt b (f a) → Exists fun c => And (Membership.mem s c) (Mem …
      ha : Membership.mem (Set.Ici a) a
      has : Membership.mem s a
      b : β
      hb : GT.gt b (f a)
      c : α
      hcs : Membership.mem s c
      hac : LT.lt (f a) (f c)
      hcb : LE.le (f c) b
      ⊢ Filter.Eventually (fun b_1 => LT.lt (f b_1) b) (nhdsWithin a (Set.Ici a))
    -/
    rw [h_mono.lt_iff_lt has hcs] at hac
    /-
      case refine_2.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : LinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      s : Set α
      a : α
      h_mono : StrictMonoOn f s
      hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
      hfs : ∀ (b : β), GT.gt b (f a) → Exists fun c => And (Membership.mem s c) (Mem …
      ha : Membership.mem (Set.Ici a) a
      has : Membership.mem s a
      b : β
      hb : GT.gt b (f a)
      c : α
      hcs : Membership.mem s c
      hac : LT.lt a c
      hcb : LE.le (f c) b
      ⊢ Filter.Eventually (fun b_1 => LT.lt (f b_1) b) (nhdsWithin a (Set.Ici a))
    -/
    filter_upwards [hs, Ico_mem_nhdsGE hac]
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : LinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      s : Set α
      a : α
      h_mono : StrictMonoOn f s
      hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
      hfs : ∀ (b : β), GT.gt b (f a) → Exists fun c => And (Membership.mem s c) (Mem …
      ha : Membership.mem (Set.Ici a) a
      has : Membership.mem s a
      b : β
      hb : GT.gt b (f a)
      c : α
      hcs : Membership.mem s c
      hac : LT.lt a c
      hcb : LE.le (f c) b
      ⊢ ∀ (a_1 : α), Membership.mem s a_1 → Membership.mem (Set.Ico a c) a_1 → LT.lt …
    -/
    rintro x hx ⟨_, hxc⟩
    /-
      case h.intro
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : LinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      s : Set α
      a : α
      h_mono : StrictMonoOn f s
      hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
      hfs : ∀ (b : β), GT.gt b (f a) → Exists fun c => And (Membership.mem s c) (Mem …
      ha : Membership.mem (Set.Ici a) a
      has : Membership.mem s a
      b : β
      hb : GT.gt b (f a)
      c : α
      hcs : Membership.mem s c
      hac : LT.lt a c
      hcb : LE.le (f c) b
      x : α
      hx : Membership.mem s x
      left✝ : LE.le a x
      hxc : LT.lt x c
      ⊢ LT.lt (f x) b
    -/
    exact ((h_mono.lt_iff_lt hx hcs).2 hxc).trans_le hcb
    /-
      🎉 no goals
    -/


/-- If `f` is a monotone function on a right neighborhood of `a` and the image of this neighborhood
under `f` meets every interval `(f a, b)`, `b > f a`, then `f` is continuous at `a` from the right.

The assumption `hfs : ∀ b > f a, ∃ c ∈ s, f c ∈ Ioo (f a) b` cannot be replaced by the weaker
assumption `hfs : ∀ b > f a, ∃ c ∈ s, f c ∈ Ioc (f a) b` we use for strictly monotone functions
because otherwise the function `ceil : ℝ → ℤ` would be a counter-example at `a = 0`. -/
theorem continuousWithinAt_right_of_monotoneOn_of_exists_between {f : α → β} {s : Set α} {a : α}
    (h_mono : MonotoneOn f s) (hs : s ∈ 𝓝[≥] a) (hfs : ∀ b > f a, ∃ c ∈ s, f c ∈ Ioo (f a) b) :
    ContinuousWithinAt f (Ici a) a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    inst✝² : LinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    s : Set α
    a : α
    h_mono : MonotoneOn f s
    hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
    hfs : ∀ (b : β), GT.gt b (f a) → Exists fun c => And (Membership.mem s c) (Mem …
    ⊢ ContinuousWithinAt f (Set.Ici a) a
  -/
  have ha : a ∈ Ici a := left_mem_Ici
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    inst✝² : LinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    s : Set α
    a : α
    h_mono : MonotoneOn f s
    hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
    hfs : ∀ (b : β), GT.gt b (f a) → Exists fun c => And (Membership.mem s c) (Mem …
    ha : Membership.mem (Set.Ici a) a
    ⊢ ContinuousWithinAt f (Set.Ici a) a
  -/
  have has : a ∈ s := mem_of_mem_nhdsWithin ha hs
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    inst✝² : LinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    s : Set α
    a : α
    h_mono : MonotoneOn f s
    hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
    hfs : ∀ (b : β), GT.gt b (f a) → Exists fun c => And (Membership.mem s c) (Mem …
    ha : Membership.mem (Set.Ici a) a
    has : Membership.mem s a
    ⊢ ContinuousWithinAt f (Set.Ici a) a
  -/
  refine tendsto_order.2 ⟨fun b hb => ?_, fun b hb => ?_⟩
  · filter_upwards [hs, @self_mem_nhdsWithin _ _ a (Ici a)] with _ hxs hxa using hb.trans_le
      (h_mono has hxs hxa)
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : LinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      s : Set α
      a : α
      h_mono : MonotoneOn f s
      hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
      hfs : ∀ (b : β), GT.gt b (f a) → Exists fun c => And (Membership.mem s c) (Mem …
      ha : Membership.mem (Set.Ici a) a
      has : Membership.mem s a
      b : β
      hb : GT.gt b (f a)
      ⊢ Filter.Eventually (fun b_1 => LT.lt (f b_1) b) (nhdsWithin a (Set.Ici a))
    -/
  · rcases hfs b hb with ⟨c, hcs, hac, hcb⟩
    /-
      case refine_2.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : LinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      s : Set α
      a : α
      h_mono : MonotoneOn f s
      hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
      hfs : ∀ (b : β), GT.gt b (f a) → Exists fun c => And (Membership.mem s c) (Mem …
      ha : Membership.mem (Set.Ici a) a
      has : Membership.mem s a
      b : β
      hb : GT.gt b (f a)
      c : α
      hcs : Membership.mem s c
      hac : LT.lt (f a) (f c)
      hcb : LT.lt (f c) b
      ⊢ Filter.Eventually (fun b_1 => LT.lt (f b_1) b) (nhdsWithin a (Set.Ici a))
    -/
    have : a < c := not_le.1 fun h => hac.not_le <| h_mono hcs has h
    /-
      case refine_2.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : LinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      s : Set α
      a : α
      h_mono : MonotoneOn f s
      hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
      hfs : ∀ (b : β), GT.gt b (f a) → Exists fun c => And (Membership.mem s c) (Mem …
      ha : Membership.mem (Set.Ici a) a
      has : Membership.mem s a
      b : β
      hb : GT.gt b (f a)
      c : α
      hcs : Membership.mem s c
      hac : LT.lt (f a) (f c)
      hcb : LT.lt (f c) b
      this : LT.lt a c
      ⊢ Filter.Eventually (fun b_1 => LT.lt (f b_1) b) (nhdsWithin a (Set.Ici a))
    -/
    filter_upwards [hs, Ico_mem_nhdsGE this]
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : LinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      s : Set α
      a : α
      h_mono : MonotoneOn f s
      hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
      hfs : ∀ (b : β), GT.gt b (f a) → Exists fun c => And (Membership.mem s c) (Mem …
      ha : Membership.mem (Set.Ici a) a
      has : Membership.mem s a
      b : β
      hb : GT.gt b (f a)
      c : α
      hcs : Membership.mem s c
      hac : LT.lt (f a) (f c)
      hcb : LT.lt (f c) b
      this : LT.lt a c
      ⊢ ∀ (a_1 : α), Membership.mem s a_1 → Membership.mem (Set.Ico a c) a_1 → LT.lt …
    -/
    rintro x hx ⟨_, hxc⟩
    /-
      case h.intro
      α : Type u_1
      β : Type u_2
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : LinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      s : Set α
      a : α
      h_mono : MonotoneOn f s
      hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
      hfs : ∀ (b : β), GT.gt b (f a) → Exists fun c => And (Membership.mem s c) (Mem …
      ha : Membership.mem (Set.Ici a) a
      has : Membership.mem s a
      b : β
      hb : GT.gt b (f a)
      c : α
      hcs : Membership.mem s c
      hac : LT.lt (f a) (f c)
      hcb : LT.lt (f c) b
      this : LT.lt a c
      x : α
      hx : Membership.mem s x
      left✝ : LE.le a x
      hxc : LT.lt x c
      ⊢ LT.lt (f x) b
    -/
    exact (h_mono hx hcs hxc.le).trans_lt hcb
    /-
      🎉 no goals
    -/


/-- If a function `f` with a densely ordered codomain is monotone on a right neighborhood of `a` and
the closure of the image of this neighborhood under `f` is a right neighborhood of `f a`, then `f`
is continuous at `a` from the right. -/
theorem continuousWithinAt_right_of_monotoneOn_of_closure_image_mem_nhdsWithin [DenselyOrdered β]
    {f : α → β} {s : Set α} {a : α} (h_mono : MonotoneOn f s) (hs : s ∈ 𝓝[≥] a)
    (hfs : closure (f '' s) ∈ 𝓝[≥] f a) : ContinuousWithinAt f (Ici a) a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : LinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : LinearOrder β
    inst✝² : TopologicalSpace β
    inst✝¹ : OrderTopology β
    inst✝ : DenselyOrdered β
    f : α → β
    s : Set α
    a : α
    h_mono : MonotoneOn f s
    hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
    hfs : Membership.mem (nhdsWithin (f a) (Set.Ici (f a))) (closure (Set.image f  …
    ⊢ ContinuousWithinAt f (Set.Ici a) a
  -/
  refine continuousWithinAt_right_of_monotoneOn_of_exists_between h_mono hs fun b hb => ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : LinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : LinearOrder β
    inst✝² : TopologicalSpace β
    inst✝¹ : OrderTopology β
    inst✝ : DenselyOrdered β
    f : α → β
    s : Set α
    a : α
    h_mono : MonotoneOn f s
    hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
    hfs : Membership.mem (nhdsWithin (f a) (Set.Ici (f a))) (closure (Set.image f  …
    b : β
    hb : GT.gt b (f a)
    ⊢ Exists fun c => And (Membership.mem s c) (Membership.mem (Set.Ioo (f a) b) ( …
  -/
  rcases (mem_nhdsGE_iff_exists_mem_Ioc_Ico_subset hb).1 hfs with ⟨b', ⟨hab', hbb'⟩, hb'⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁶ : LinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : LinearOrder β
    inst✝² : TopologicalSpace β
    inst✝¹ : OrderTopology β
    inst✝ : DenselyOrdered β
    f : α → β
    s : Set α
    a : α
    h_mono : MonotoneOn f s
    hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
    hfs : Membership.mem (nhdsWithin (f a) (Set.Ici (f a))) (closure (Set.image f  …
    b : β
    hb : GT.gt b (f a)
    b' : β
    hb' : HasSubset.Subset (Set.Ico (f a) b') (closure (Set.image f s))
    hab' : LT.lt (f a) b'
    hbb' : LE.le b' b
    ⊢ Exists fun c => And (Membership.mem s c) (Membership.mem (Set.Ioo (f a) b) ( …
  -/
  rcases exists_between hab' with ⟨c', hc'⟩
  rcases mem_closure_iff.1 (hb' ⟨hc'.1.le, hc'.2⟩) (Ioo (f a) b') isOpen_Ioo hc' with
    ⟨_, hc, ⟨c, hcs, rfl⟩⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝⁶ : LinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : LinearOrder β
    inst✝² : TopologicalSpace β
    inst✝¹ : OrderTopology β
    inst✝ : DenselyOrdered β
    f : α → β
    s : Set α
    a : α
    h_mono : MonotoneOn f s
    hs : Membership.mem (nhdsWithin a (Set.Ici a)) s
    hfs : Membership.mem (nhdsWithin (f a) (Set.Ici (f a))) (closure (Set.image f  …
    b : β
    hb : GT.gt b (f a)
    b' : β
    hb' : HasSubset.Subset (Set.Ico (f a) b') (closure (Set.image f s))
    hab' : LT.lt (f a) b'
    hbb' : LE.le b' b
    c' : β
    hc' : And (LT.lt (f a) c') (LT.lt c' b')
    c : α
    hcs : Membership.mem s c
    hc : Membership.mem (Set.Ioo (f a) b') (f c)
    ⊢ Exists fun c => And (Membership.mem s c) (Membership.mem (Set.Ioo (f a) b) ( …
  -/
  exact ⟨c, hcs, hc.1, hc.2.trans_le hbb'⟩
  /-
    🎉 no goals
  -/


/-- If a function `f` with a densely ordered codomain is monotone on a right neighborhood of `a` and
the image of this neighborhood under `f` is a right neighborhood of `f a`, then `f` is continuous at
`a` from the right. -/
theorem continuousWithinAt_right_of_monotoneOn_of_image_mem_nhdsWithin [DenselyOrdered β]
    {f : α → β} {s : Set α} {a : α} (h_mono : MonotoneOn f s) (hs : s ∈ 𝓝[≥] a)
    (hfs : f '' s ∈ 𝓝[≥] f a) : ContinuousWithinAt f (Ici a) a :=
  continuousWithinAt_right_of_monotoneOn_of_closure_image_mem_nhdsWithin h_mono hs <|
    mem_of_superset hfs subset_closure


/-- If a function `f` with a densely ordered codomain is strictly monotone on a right neighborhood
of `a` and the closure of the image of this neighborhood under `f` is a right neighborhood of `f a`,
then `f` is continuous at `a` from the right. -/
theorem StrictMonoOn.continuousWithinAt_right_of_closure_image_mem_nhdsWithin [DenselyOrdered β]
    {f : α → β} {s : Set α} {a : α} (h_mono : StrictMonoOn f s) (hs : s ∈ 𝓝[≥] a)
    (hfs : closure (f '' s) ∈ 𝓝[≥] f a) : ContinuousWithinAt f (Ici a) a :=
  continuousWithinAt_right_of_monotoneOn_of_closure_image_mem_nhdsWithin
    (fun _ hx _ hy => (h_mono.le_iff_le hx hy).2) hs hfs


/-- If a function `f` with a densely ordered codomain is strictly monotone on a right neighborhood
of `a` and the image of this neighborhood under `f` is a right neighborhood of `f a`, then `f` is
continuous at `a` from the right. -/
theorem StrictMonoOn.continuousWithinAt_right_of_image_mem_nhdsWithin [DenselyOrdered β] {f : α → β}
    {s : Set α} {a : α} (h_mono : StrictMonoOn f s) (hs : s ∈ 𝓝[≥] a) (hfs : f '' s ∈ 𝓝[≥] f a) :
    ContinuousWithinAt f (Ici a) a :=
  h_mono.continuousWithinAt_right_of_closure_image_mem_nhdsWithin hs
    (mem_of_superset hfs subset_closure)


/-- If a function `f` is strictly monotone on a right neighborhood of `a` and the image of this
neighborhood under `f` includes `Ioi (f a)`, then `f` is continuous at `a` from the right. -/
theorem StrictMonoOn.continuousWithinAt_right_of_surjOn {f : α → β} {s : Set α} {a : α}
    (h_mono : StrictMonoOn f s) (hs : s ∈ 𝓝[≥] a) (hfs : SurjOn f s (Ioi (f a))) :
    ContinuousWithinAt f (Ici a) a :=
  h_mono.continuousWithinAt_right_of_exists_between hs fun _ hb =>
    let ⟨c, hcs, hcb⟩ := hfs hb
    ⟨c, hcs, hcb.symm ▸ hb, hcb.le⟩


/-- If `f` is a strictly monotone function on a left neighborhood of `a` and the image of this
neighborhood under `f` meets every interval `[b, f a)`, `b < f a`, then `f` is continuous at `a`
from the left.

The assumption `hfs : ∀ b < f a, ∃ c ∈ s, f c ∈ Ico b (f a)` is required because otherwise the
function `f : ℝ → ℝ` given by `f x = if x < 0 then x else x + 1` would be a counter-example at
`a = 0`. -/
theorem StrictMonoOn.continuousWithinAt_left_of_exists_between {f : α → β} {s : Set α} {a : α}
    (h_mono : StrictMonoOn f s) (hs : s ∈ 𝓝[≤] a) (hfs : ∀ b < f a, ∃ c ∈ s, f c ∈ Ico b (f a)) :
    ContinuousWithinAt f (Iic a) a :=
  h_mono.dual.continuousWithinAt_right_of_exists_between hs fun b hb =>
    let ⟨c, hcs, hcb, hca⟩ := hfs b hb
    ⟨c, hcs, hca, hcb⟩


/-- If `f` is a monotone function on a left neighborhood of `a` and the image of this neighborhood
under `f` meets every interval `(b, f a)`, `b < f a`, then `f` is continuous at `a` from the left.

The assumption `hfs : ∀ b < f a, ∃ c ∈ s, f c ∈ Ioo b (f a)` cannot be replaced by the weaker
assumption `hfs : ∀ b < f a, ∃ c ∈ s, f c ∈ Ico b (f a)` we use for strictly monotone functions
because otherwise the function `floor : ℝ → ℤ` would be a counter-example at `a = 0`. -/
theorem continuousWithinAt_left_of_monotoneOn_of_exists_between {f : α → β} {s : Set α} {a : α}
    (hf : MonotoneOn f s) (hs : s ∈ 𝓝[≤] a) (hfs : ∀ b < f a, ∃ c ∈ s, f c ∈ Ioo b (f a)) :
    ContinuousWithinAt f (Iic a) a :=
  @continuousWithinAt_right_of_monotoneOn_of_exists_between αᵒᵈ βᵒᵈ _ _ _ _ _ _ f s a hf.dual hs
    fun b hb =>
    let ⟨c, hcs, hcb, hca⟩ := hfs b hb
    ⟨c, hcs, hca, hcb⟩


/-- If a function `f` with a densely ordered codomain is monotone on a left neighborhood of `a` and
the closure of the image of this neighborhood under `f` is a left neighborhood of `f a`, then `f` is
continuous at `a` from the left -/
theorem continuousWithinAt_left_of_monotoneOn_of_closure_image_mem_nhdsWithin [DenselyOrdered β]
    {f : α → β} {s : Set α} {a : α} (hf : MonotoneOn f s) (hs : s ∈ 𝓝[≤] a)
    (hfs : closure (f '' s) ∈ 𝓝[≤] f a) : ContinuousWithinAt f (Iic a) a :=
  @continuousWithinAt_right_of_monotoneOn_of_closure_image_mem_nhdsWithin αᵒᵈ βᵒᵈ _ _ _ _ _ _ _ f s
    a hf.dual hs hfs


/-- If a function `f` with a densely ordered codomain is monotone on a left neighborhood of `a` and
the image of this neighborhood under `f` is a left neighborhood of `f a`, then `f` is continuous at
`a` from the left. -/
theorem continuousWithinAt_left_of_monotoneOn_of_image_mem_nhdsWithin [DenselyOrdered β] {f : α → β}
    {s : Set α} {a : α} (h_mono : MonotoneOn f s) (hs : s ∈ 𝓝[≤] a) (hfs : f '' s ∈ 𝓝[≤] f a) :
    ContinuousWithinAt f (Iic a) a :=
  continuousWithinAt_left_of_monotoneOn_of_closure_image_mem_nhdsWithin h_mono hs
    (mem_of_superset hfs subset_closure)


/-- If a function `f` with a densely ordered codomain is strictly monotone on a left neighborhood of
`a` and the closure of the image of this neighborhood under `f` is a left neighborhood of `f a`,
then `f` is continuous at `a` from the left. -/
theorem StrictMonoOn.continuousWithinAt_left_of_closure_image_mem_nhdsWithin [DenselyOrdered β]
    {f : α → β} {s : Set α} {a : α} (h_mono : StrictMonoOn f s) (hs : s ∈ 𝓝[≤] a)
    (hfs : closure (f '' s) ∈ 𝓝[≤] f a) : ContinuousWithinAt f (Iic a) a :=
  h_mono.dual.continuousWithinAt_right_of_closure_image_mem_nhdsWithin hs hfs


/-- If a function `f` with a densely ordered codomain is strictly monotone on a left neighborhood of
`a` and the image of this neighborhood under `f` is a left neighborhood of `f a`, then `f` is
continuous at `a` from the left. -/
theorem StrictMonoOn.continuousWithinAt_left_of_image_mem_nhdsWithin [DenselyOrdered β] {f : α → β}
    {s : Set α} {a : α} (h_mono : StrictMonoOn f s) (hs : s ∈ 𝓝[≤] a) (hfs : f '' s ∈ 𝓝[≤] f a) :
    ContinuousWithinAt f (Iic a) a :=
  h_mono.dual.continuousWithinAt_right_of_image_mem_nhdsWithin hs hfs


/-- If a function `f` is strictly monotone on a left neighborhood of `a` and the image of this
neighborhood under `f` includes `Iio (f a)`, then `f` is continuous at `a` from the left. -/
theorem StrictMonoOn.continuousWithinAt_left_of_surjOn {f : α → β} {s : Set α} {a : α}
    (h_mono : StrictMonoOn f s) (hs : s ∈ 𝓝[≤] a) (hfs : SurjOn f s (Iio (f a))) :
    ContinuousWithinAt f (Iic a) a :=
  h_mono.dual.continuousWithinAt_right_of_surjOn hs hfs


/-- If a function `f` is strictly monotone on a neighborhood of `a` and the image of this
neighborhood under `f` meets every interval `[b, f a)`, `b < f a`, and every interval
`(f a, b]`, `b > f a`, then `f` is continuous at `a`. -/
theorem StrictMonoOn.continuousAt_of_exists_between {f : α → β} {s : Set α} {a : α}
    (h_mono : StrictMonoOn f s) (hs : s ∈ 𝓝 a) (hfs_l : ∀ b < f a, ∃ c ∈ s, f c ∈ Ico b (f a))
    (hfs_r : ∀ b > f a, ∃ c ∈ s, f c ∈ Ioc (f a) b) : ContinuousAt f a :=
  continuousAt_iff_continuous_left_right.2
    ⟨h_mono.continuousWithinAt_left_of_exists_between (mem_nhdsWithin_of_mem_nhds hs) hfs_l,
      h_mono.continuousWithinAt_right_of_exists_between (mem_nhdsWithin_of_mem_nhds hs) hfs_r⟩


/-- If a function `f` with a densely ordered codomain is strictly monotone on a neighborhood of `a`
and the closure of the image of this neighborhood under `f` is a neighborhood of `f a`, then `f` is
continuous at `a`. -/
theorem StrictMonoOn.continuousAt_of_closure_image_mem_nhds [DenselyOrdered β] {f : α → β}
    {s : Set α} {a : α} (h_mono : StrictMonoOn f s) (hs : s ∈ 𝓝 a)
    (hfs : closure (f '' s) ∈ 𝓝 (f a)) : ContinuousAt f a :=
  continuousAt_iff_continuous_left_right.2
    ⟨h_mono.continuousWithinAt_left_of_closure_image_mem_nhdsWithin (mem_nhdsWithin_of_mem_nhds hs)
        (mem_nhdsWithin_of_mem_nhds hfs),
      h_mono.continuousWithinAt_right_of_closure_image_mem_nhdsWithin
        (mem_nhdsWithin_of_mem_nhds hs) (mem_nhdsWithin_of_mem_nhds hfs)⟩


/-- If a function `f` with a densely ordered codomain is strictly monotone on a neighborhood of `a`
and the image of this set under `f` is a neighborhood of `f a`, then `f` is continuous at `a`. -/
theorem StrictMonoOn.continuousAt_of_image_mem_nhds [DenselyOrdered β] {f : α → β} {s : Set α}
    {a : α} (h_mono : StrictMonoOn f s) (hs : s ∈ 𝓝 a) (hfs : f '' s ∈ 𝓝 (f a)) :
    ContinuousAt f a :=
  h_mono.continuousAt_of_closure_image_mem_nhds hs (mem_of_superset hfs subset_closure)


/-- If `f` is a monotone function on a neighborhood of `a` and the image of this neighborhood under
`f` meets every interval `(b, f a)`, `b < f a`, and every interval `(f a, b)`, `b > f a`, then `f`
is continuous at `a`. -/
theorem continuousAt_of_monotoneOn_of_exists_between {f : α → β} {s : Set α} {a : α}
    (h_mono : MonotoneOn f s) (hs : s ∈ 𝓝 a) (hfs_l : ∀ b < f a, ∃ c ∈ s, f c ∈ Ioo b (f a))
    (hfs_r : ∀ b > f a, ∃ c ∈ s, f c ∈ Ioo (f a) b) : ContinuousAt f a :=
  continuousAt_iff_continuous_left_right.2
    ⟨continuousWithinAt_left_of_monotoneOn_of_exists_between h_mono (mem_nhdsWithin_of_mem_nhds hs)
        hfs_l,
      continuousWithinAt_right_of_monotoneOn_of_exists_between h_mono
        (mem_nhdsWithin_of_mem_nhds hs) hfs_r⟩


/-- If a function `f` with a densely ordered codomain is monotone on a neighborhood of `a` and the
closure of the image of this neighborhood under `f` is a neighborhood of `f a`, then `f` is
continuous at `a`. -/
theorem continuousAt_of_monotoneOn_of_closure_image_mem_nhds [DenselyOrdered β] {f : α → β}
    {s : Set α} {a : α} (h_mono : MonotoneOn f s) (hs : s ∈ 𝓝 a)
    (hfs : closure (f '' s) ∈ 𝓝 (f a)) : ContinuousAt f a :=
  continuousAt_iff_continuous_left_right.2
    ⟨continuousWithinAt_left_of_monotoneOn_of_closure_image_mem_nhdsWithin h_mono
        (mem_nhdsWithin_of_mem_nhds hs) (mem_nhdsWithin_of_mem_nhds hfs),
      continuousWithinAt_right_of_monotoneOn_of_closure_image_mem_nhdsWithin h_mono
        (mem_nhdsWithin_of_mem_nhds hs) (mem_nhdsWithin_of_mem_nhds hfs)⟩


/-- If a function `f` with a densely ordered codomain is monotone on a neighborhood of `a` and the
image of this neighborhood under `f` is a neighborhood of `f a`, then `f` is continuous at `a`. -/
theorem continuousAt_of_monotoneOn_of_image_mem_nhds [DenselyOrdered β] {f : α → β} {s : Set α}
    {a : α} (h_mono : MonotoneOn f s) (hs : s ∈ 𝓝 a) (hfs : f '' s ∈ 𝓝 (f a)) : ContinuousAt f a :=
  continuousAt_of_monotoneOn_of_closure_image_mem_nhds h_mono hs
    (mem_of_superset hfs subset_closure)


/-- A monotone function with densely ordered codomain and a dense range is continuous. -/
theorem Monotone.continuous_of_denseRange [DenselyOrdered β] {f : α → β} (h_mono : Monotone f)
    (h_dense : DenseRange f) : Continuous f :=
  continuous_iff_continuousAt.mpr fun a =>
    continuousAt_of_monotoneOn_of_closure_image_mem_nhds (fun _ _ _ _ hxy => h_mono hxy)
        univ_mem <|
         /-
           α : Type u_1
           β : Type u_2
           inst✝⁶ : LinearOrder α
           inst✝⁵ : TopologicalSpace α
           inst✝⁴ : OrderTopology α
           inst✝³ : LinearOrder β
           inst✝² : TopologicalSpace β
           inst✝¹ : OrderTopology β
           inst✝ : DenselyOrdered β
           f : α → β
           h_mono : Monotone f
           h_dense : DenseRange f
           a : α
           ⊢ Membership.mem (nhds (f a)) (closure (Set.image f Set.univ))
         -/
      by simp only [image_univ, h_dense.closure_eq, univ_mem]
         /-
           🎉 no goals
         -/


/-- A monotone surjective function with a densely ordered codomain is continuous. -/
theorem Monotone.continuous_of_surjective [DenselyOrdered β] {f : α → β} (h_mono : Monotone f)
    (h_surj : Function.Surjective f) : Continuous f :=
  h_mono.continuous_of_denseRange h_surj.denseRange


protected theorem continuous (e : α ≃o β) : Continuous e := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : PartialOrder α
    inst✝⁴ : PartialOrder β
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : OrderTopology α
    inst✝ : OrderTopology β
    e : OrderIso α β
    ⊢ Continuous ⇑e
  -/
  rw [‹OrderTopology β›.topology_eq_generate_intervals, continuous_generateFrom_iff]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : PartialOrder α
    inst✝⁴ : PartialOrder β
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    inst✝¹ : OrderTopology α
    inst✝ : OrderTopology β
    e : OrderIso α β
    ⊢ ∀ (s : Set β), Membership.mem (setOf fun s => Exists fun a => Or (Eq s (Set. …
  -/
  rintro s ⟨a, rfl | rfl⟩
    /-
      case intro.inl
      α : Type u_1
      β : Type u_2
      inst✝⁵ : PartialOrder α
      inst✝⁴ : PartialOrder β
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : OrderTopology α
      inst✝ : OrderTopology β
      e : OrderIso α β
      a : β
      ⊢ IsOpen (Set.preimage (⇑e) (Set.Ioi a))
    -/
  · rw [e.preimage_Ioi]
    /-
      case intro.inl
      α : Type u_1
      β : Type u_2
      inst✝⁵ : PartialOrder α
      inst✝⁴ : PartialOrder β
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : OrderTopology α
      inst✝ : OrderTopology β
      e : OrderIso α β
      a : β
      ⊢ IsOpen (Set.Ioi (e.symm a))
    -/
    apply isOpen_lt'
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      α : Type u_1
      β : Type u_2
      inst✝⁵ : PartialOrder α
      inst✝⁴ : PartialOrder β
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : OrderTopology α
      inst✝ : OrderTopology β
      e : OrderIso α β
      a : β
      ⊢ IsOpen (Set.preimage (⇑e) (Set.Iio a))
    -/
  · rw [e.preimage_Iio]
    /-
      case intro.inr
      α : Type u_1
      β : Type u_2
      inst✝⁵ : PartialOrder α
      inst✝⁴ : PartialOrder β
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : OrderTopology α
      inst✝ : OrderTopology β
      e : OrderIso α β
      a : β
      ⊢ IsOpen (Set.Iio (e.symm a))
    -/
    apply isOpen_gt'
    /-
      🎉 no goals
    -/


/-- An order isomorphism between two linear order `OrderTopology` spaces is a homeomorphism. -/
def toHomeomorph (e : α ≃o β) : α ≃ₜ β :=
  { e with
    continuous_toFun := e.continuous
    continuous_invFun := e.symm.continuous }


@[simp]
theorem coe_toHomeomorph (e : α ≃o β) : ⇑e.toHomeomorph = e :=
  rfl


@[simp]
theorem coe_toHomeomorph_symm (e : α ≃o β) : ⇑e.toHomeomorph.symm = e.symm :=
  rfl



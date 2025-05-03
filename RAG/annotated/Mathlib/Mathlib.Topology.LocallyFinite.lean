/-- A family of sets in `Set X` is locally finite if at every point `x : X`,
there is a neighborhood of `x` which meets only finitely many sets in the family. -/
def LocallyFinite (f : ι → Set X) :=
  ∀ x : X, ∃ t ∈ 𝓝 x, { i | (f i ∩ t).Nonempty }.Finite


theorem locallyFinite_of_finite [Finite ι] (f : ι → Set X) : LocallyFinite f := fun _ =>
  ⟨univ, univ_mem, toFinite _⟩


theorem point_finite (hf : LocallyFinite f) (x : X) : { b | x ∈ f b }.Finite :=
  let ⟨_t, hxt, ht⟩ := hf x
  ht.subset fun _b hb => ⟨x, hb, mem_of_mem_nhds hxt⟩


protected theorem subset (hf : LocallyFinite f) (hg : ∀ i, g i ⊆ f i) : LocallyFinite g := fun a =>
  let ⟨t, ht₁, ht₂⟩ := hf a
  ⟨t, ht₁, ht₂.subset fun i hi => hi.mono <| inter_subset_inter (hg i) Subset.rfl⟩


theorem comp_injOn {g : ι' → ι} (hf : LocallyFinite f) (hg : InjOn g { i | (f (g i)).Nonempty }) :
    LocallyFinite (f ∘ g) := fun x => by
  /-
    ι : Type u_1
    ι' : Type u_2
    X : Type u_4
    inst✝ : TopologicalSpace X
    f : ι → Set X
    g : ι' → ι
    hf : LocallyFinite f
    hg : Set.InjOn g (setOf fun i => (f (g i)).Nonempty)
    x : X
    ⊢ Exists fun t => And (Membership.mem (nhds x) t) (setOf fun i => (Inter.inter …
  -/
  let ⟨t, htx, htf⟩ := hf x
  /-
    ι : Type u_1
    ι' : Type u_2
    X : Type u_4
    inst✝ : TopologicalSpace X
    f : ι → Set X
    g : ι' → ι
    hf : LocallyFinite f
    hg : Set.InjOn g (setOf fun i => (f (g i)).Nonempty)
    x : X
    t : Set X
    htx : Membership.mem (nhds x) t
    htf : (setOf fun i => (Inter.inter (f i) t).Nonempty).Finite
    ⊢ Exists fun t => And (Membership.mem (nhds x) t) (setOf fun i => (Inter.inter …
  -/
  refine ⟨t, htx, htf.preimage <| ?_⟩
  /-
    ι : Type u_1
    ι' : Type u_2
    X : Type u_4
    inst✝ : TopologicalSpace X
    f : ι → Set X
    g : ι' → ι
    hf : LocallyFinite f
    hg : Set.InjOn g (setOf fun i => (f (g i)).Nonempty)
    x : X
    t : Set X
    htx : Membership.mem (nhds x) t
    htf : (setOf fun i => (Inter.inter (f i) t).Nonempty).Finite
    ⊢ Set.InjOn g (Set.preimage g (setOf fun i => (Inter.inter (f i) t).Nonempty))
  -/
  exact hg.mono fun i (hi : Set.Nonempty _) => hi.left
  /-
    🎉 no goals
  -/


theorem comp_injective {g : ι' → ι} (hf : LocallyFinite f) (hg : Injective g) :
    LocallyFinite (f ∘ g) :=
  hf.comp_injOn hg.injOn


theorem _root_.locallyFinite_iff_smallSets :
    LocallyFinite f ↔ ∀ x, ∀ᶠ s in (𝓝 x).smallSets, { i | (f i ∩ s).Nonempty }.Finite :=
  forall_congr' fun _ => Iff.symm <|
    eventually_smallSets' fun _s _t hst ht =>
      ht.subset fun _i hi => hi.mono <| inter_subset_inter_right _ hst


protected theorem eventually_smallSets (hf : LocallyFinite f) (x : X) :
    ∀ᶠ s in (𝓝 x).smallSets, { i | (f i ∩ s).Nonempty }.Finite :=
  locallyFinite_iff_smallSets.mp hf x


theorem exists_mem_basis {ι' : Sort*} (hf : LocallyFinite f) {p : ι' → Prop} {s : ι' → Set X}
    {x : X} (hb : (𝓝 x).HasBasis p s) : ∃ i, p i ∧ { j | (f j ∩ s i).Nonempty }.Finite :=
  let ⟨i, hpi, hi⟩ := hb.smallSets.eventually_iff.mp (hf.eventually_smallSets x)
  ⟨i, hpi, hi Subset.rfl⟩


protected theorem nhdsWithin_iUnion (hf : LocallyFinite f) (a : X) :
    𝓝[⋃ i, f i] a = ⨆ i, 𝓝[f i] a := by
  /-
    ι : Type u_1
    X : Type u_4
    inst✝ : TopologicalSpace X
    f : ι → Set X
    hf : LocallyFinite f
    a : X
    ⊢ Eq (nhdsWithin a (Set.iUnion fun i => f i)) (iSup fun i => nhdsWithin a (f i))
  -/
  rcases hf a with ⟨U, haU, hfin⟩
  /-
    case intro.intro
    ι : Type u_1
    X : Type u_4
    inst✝ : TopologicalSpace X
    f : ι → Set X
    hf : LocallyFinite f
    a : X
    U : Set X
    haU : Membership.mem (nhds a) U
    hfin : (setOf fun i => (Inter.inter (f i) U).Nonempty).Finite
    ⊢ Eq (nhdsWithin a (Set.iUnion fun i => f i)) (iSup fun i => nhdsWithin a (f i))
  -/
  refine le_antisymm ?_ (Monotone.le_map_iSup fun _ _ ↦ nhdsWithin_mono _)
  calc
    𝓝[⋃ i, f i] a = 𝓝[⋃ i, f i ∩ U] a := by
      rw [← iUnion_inter, ← nhdsWithin_inter_of_mem' (nhdsWithin_le_nhds haU)]
    _ = 𝓝[⋃ i ∈ {j | (f j ∩ U).Nonempty}, (f i ∩ U)] a := by
      simp only [mem_setOf_eq, iUnion_nonempty_self]
    _ = ⨆ i ∈ {j | (f j ∩ U).Nonempty}, 𝓝[f i ∩ U] a := nhdsWithin_biUnion hfin _ _
    _ ≤ ⨆ i, 𝓝[f i ∩ U] a := iSup₂_le_iSup _ _
    _ ≤ ⨆ i, 𝓝[f i] a := iSup_mono fun i ↦ nhdsWithin_mono _ inter_subset_left


theorem continuousOn_iUnion' {g : X → Y} (hf : LocallyFinite f)
    (hc : ∀ i x, x ∈ closure (f i) → ContinuousWithinAt g (f i) x) :
    ContinuousOn g (⋃ i, f i) := by
  /-
    ι : Type u_1
    X : Type u_4
    Y : Type u_5
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : ι → Set X
    g : X → Y
    hf : LocallyFinite f
    hc : ∀ (i : ι) (x : X), Membership.mem (closure (f i)) x → ContinuousWithinAt  …
    ⊢ ContinuousOn g (Set.iUnion fun i => f i)
  -/
  rintro x -
  /-
    ι : Type u_1
    X : Type u_4
    Y : Type u_5
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : ι → Set X
    g : X → Y
    hf : LocallyFinite f
    hc : ∀ (i : ι) (x : X), Membership.mem (closure (f i)) x → ContinuousWithinAt  …
    x : X
    ⊢ ContinuousWithinAt g (Set.iUnion fun i => f i) x
  -/
  rw [ContinuousWithinAt, hf.nhdsWithin_iUnion, tendsto_iSup]
  /-
    ι : Type u_1
    X : Type u_4
    Y : Type u_5
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : ι → Set X
    g : X → Y
    hf : LocallyFinite f
    hc : ∀ (i : ι) (x : X), Membership.mem (closure (f i)) x → ContinuousWithinAt  …
    x : X
    ⊢ ∀ (i : ι), Filter.Tendsto g (nhdsWithin x (f i)) (nhds (g x))
  -/
  intro i
  /-
    ι : Type u_1
    X : Type u_4
    Y : Type u_5
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : ι → Set X
    g : X → Y
    hf : LocallyFinite f
    hc : ∀ (i : ι) (x : X), Membership.mem (closure (f i)) x → ContinuousWithinAt  …
    x : X
    i : ι
    ⊢ Filter.Tendsto g (nhdsWithin x (f i)) (nhds (g x))
  -/
  by_cases hx : x ∈ closure (f i)
    /-
      case pos
      ι : Type u_1
      X : Type u_4
      Y : Type u_5
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : ι → Set X
      g : X → Y
      hf : LocallyFinite f
      hc : ∀ (i : ι) (x : X), Membership.mem (closure (f i)) x → ContinuousWithinAt  …
      x : X
      i : ι
      hx : Membership.mem (closure (f i)) x
      ⊢ Filter.Tendsto g (nhdsWithin x (f i)) (nhds (g x))
    -/
  · exact hc i _ hx
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      X : Type u_4
      Y : Type u_5
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : ι → Set X
      g : X → Y
      hf : LocallyFinite f
      hc : ∀ (i : ι) (x : X), Membership.mem (closure (f i)) x → ContinuousWithinAt  …
      x : X
      i : ι
      hx : Not (Membership.mem (closure (f i)) x)
      ⊢ Filter.Tendsto g (nhdsWithin x (f i)) (nhds (g x))
    -/
  · rw [mem_closure_iff_nhdsWithin_neBot, not_neBot] at hx
    /-
      case neg
      ι : Type u_1
      X : Type u_4
      Y : Type u_5
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : ι → Set X
      g : X → Y
      hf : LocallyFinite f
      hc : ∀ (i : ι) (x : X), Membership.mem (closure (f i)) x → ContinuousWithinAt  …
      x : X
      i : ι
      hx : Eq (nhdsWithin x (f i)) Bot.bot
      ⊢ Filter.Tendsto g (nhdsWithin x (f i)) (nhds (g x))
    -/
    rw [hx]
    /-
      case neg
      ι : Type u_1
      X : Type u_4
      Y : Type u_5
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : ι → Set X
      g : X → Y
      hf : LocallyFinite f
      hc : ∀ (i : ι) (x : X), Membership.mem (closure (f i)) x → ContinuousWithinAt  …
      x : X
      i : ι
      hx : Eq (nhdsWithin x (f i)) Bot.bot
      ⊢ Filter.Tendsto g Bot.bot (nhds (g x))
    -/
    exact tendsto_bot
    /-
      🎉 no goals
    -/


theorem continuousOn_iUnion {g : X → Y} (hf : LocallyFinite f) (h_cl : ∀ i, IsClosed (f i))
    (h_cont : ∀ i, ContinuousOn g (f i)) : ContinuousOn g (⋃ i, f i) :=
  hf.continuousOn_iUnion' fun i x hx ↦ h_cont i x <| (h_cl i).closure_subset hx


protected theorem continuous' {g : X → Y} (hf : LocallyFinite f) (h_cov : ⋃ i, f i = univ)
    (hc : ∀ i x, x ∈ closure (f i) → ContinuousWithinAt g (f i) x) :
    Continuous g :=
  continuous_iff_continuousOn_univ.2 <| h_cov ▸ hf.continuousOn_iUnion' hc


protected theorem continuous {g : X → Y} (hf : LocallyFinite f) (h_cov : ⋃ i, f i = univ)
    (h_cl : ∀ i, IsClosed (f i)) (h_cont : ∀ i, ContinuousOn g (f i)) :
    Continuous g :=
  continuous_iff_continuousOn_univ.2 <| h_cov ▸ hf.continuousOn_iUnion h_cl h_cont


protected theorem closure (hf : LocallyFinite f) : LocallyFinite fun i => closure (f i) := by
  /-
    ι : Type u_1
    X : Type u_4
    inst✝ : TopologicalSpace X
    f : ι → Set X
    hf : LocallyFinite f
    ⊢ LocallyFinite fun i => closure (f i)
  -/
  intro x
  /-
    ι : Type u_1
    X : Type u_4
    inst✝ : TopologicalSpace X
    f : ι → Set X
    hf : LocallyFinite f
    x : X
    ⊢ Exists fun t => And (Membership.mem (nhds x) t) (setOf fun i => (Inter.inter …
  -/
  rcases hf x with ⟨s, hsx, hsf⟩
  /-
    case intro.intro
    ι : Type u_1
    X : Type u_4
    inst✝ : TopologicalSpace X
    f : ι → Set X
    hf : LocallyFinite f
    x : X
    s : Set X
    hsx : Membership.mem (nhds x) s
    hsf : (setOf fun i => (Inter.inter (f i) s).Nonempty).Finite
    ⊢ Exists fun t => And (Membership.mem (nhds x) t) (setOf fun i => (Inter.inter …
  -/
  refine ⟨interior s, interior_mem_nhds.2 hsx, hsf.subset fun i hi => ?_⟩
  exact (hi.mono isOpen_interior.closure_inter).of_closure.mono
    (inter_subset_inter_right _ interior_subset)


theorem closure_iUnion (h : LocallyFinite f) : closure (⋃ i, f i) = ⋃ i, closure (f i) := by
  /-
    ι : Type u_1
    X : Type u_4
    inst✝ : TopologicalSpace X
    f : ι → Set X
    h : LocallyFinite f
    ⊢ Eq (closure (Set.iUnion fun i => f i)) (Set.iUnion fun i => closure (f i))
  -/
  ext x
  /-
    case h
    ι : Type u_1
    X : Type u_4
    inst✝ : TopologicalSpace X
    f : ι → Set X
    h : LocallyFinite f
    x : X
    ⊢ Iff (Membership.mem (closure (Set.iUnion fun i => f i)) x) (Membership.mem ( …
  -/
  simp only [mem_closure_iff_nhdsWithin_neBot, h.nhdsWithin_iUnion, iSup_neBot, mem_iUnion]
  /-
    🎉 no goals
  -/


theorem isClosed_iUnion (hf : LocallyFinite f) (hc : ∀ i, IsClosed (f i)) :
    IsClosed (⋃ i, f i) := by
  /-
    ι : Type u_1
    X : Type u_4
    inst✝ : TopologicalSpace X
    f : ι → Set X
    hf : LocallyFinite f
    hc : ∀ (i : ι), IsClosed (f i)
    ⊢ IsClosed (Set.iUnion fun i => f i)
  -/
  simp only [← closure_eq_iff_isClosed, hf.closure_iUnion, (hc _).closure_eq]
  /-
    🎉 no goals
  -/


/-- If `f : β → Set α` is a locally finite family of closed sets, then for any `x : α`, the
intersection of the complements to `f i`, `x ∉ f i`, is a neighbourhood of `x`. -/
theorem iInter_compl_mem_nhds (hf : LocallyFinite f) (hc : ∀ i, IsClosed (f i)) (x : X) :
    (⋂ (i) (_ : x ∉ f i), (f i)ᶜ) ∈ 𝓝 x := by
  /-
    ι : Type u_1
    X : Type u_4
    inst✝ : TopologicalSpace X
    f : ι → Set X
    hf : LocallyFinite f
    hc : ∀ (i : ι), IsClosed (f i)
    x : X
    ⊢ Membership.mem (nhds x) (Set.iInter fun i => Set.iInter fun x => HasCompl.co …
  -/
  refine IsOpen.mem_nhds ?_ (mem_iInter₂.2 fun i => id)
  suffices IsClosed (⋃ i : { i // x ∉ f i }, f i) by
    rwa [← isOpen_compl_iff, compl_iUnion, iInter_subtype] at this
  /-
    ι : Type u_1
    X : Type u_4
    inst✝ : TopologicalSpace X
    f : ι → Set X
    hf : LocallyFinite f
    hc : ∀ (i : ι), IsClosed (f i)
    x : X
    ⊢ IsClosed (Set.iUnion fun i => f ↑i)
  -/
  exact (hf.comp_injective Subtype.val_injective).isClosed_iUnion fun i => hc _
  /-
    🎉 no goals
  -/


/-- Let `f : ℕ → Π a, β a` be a sequence of (dependent) functions on a topological space. Suppose
that the family of sets `s n = {x | f (n + 1) x ≠ f n x}` is locally finite. Then there exists a
function `F : Π a, β a` such that for any `x`, we have `f n x = F x` on the product of an infinite
interval `[N, +∞)` and a neighbourhood of `x`.

We formulate the conclusion in terms of the product of filter `Filter.atTop` and `𝓝 x`. -/
theorem exists_forall_eventually_eq_prod {π : X → Sort*} {f : ℕ → ∀ x : X, π x}
    (hf : LocallyFinite fun n => { x | f (n + 1) x ≠ f n x }) :
    ∃ F : ∀ x : X, π x, ∀ x, ∀ᶠ p : ℕ × X in atTop ×ˢ 𝓝 x, f p.1 p.2 = F p.2 := by
  /-
    X : Type u_4
    inst✝ : TopologicalSpace X
    π : X → Sort u_6
    f : Nat → (x : X) → π x
    hf : LocallyFinite fun n => setOf fun x => Ne (f (HAdd.hAdd n 1) x) (f n x)
    ⊢ Exists fun F => ∀ (x : X), Filter.Eventually (fun p => Eq (f p.1 p.2) (F p.2 …
  -/
  choose U hUx hU using hf
  /-
    X : Type u_4
    inst✝ : TopologicalSpace X
    π : X → Sort u_6
    f : Nat → (x : X) → π x
    U : X → Set X
    hUx : ∀ (x : X), Membership.mem (nhds x) (U x)
    hU : ∀ (x : X), (setOf fun i => (Inter.inter ((fun n => setOf fun x => Ne (f ( …
    ⊢ Exists fun F => ∀ (x : X), Filter.Eventually (fun p => Eq (f p.1 p.2) (F p.2 …
  -/
  choose N hN using fun x => (hU x).bddAbove
  replace hN : ∀ (x), ∀ n > N x, ∀ y ∈ U x, f (n + 1) y = f n y :=
    fun x n hn y hy => by_contra fun hne => hn.lt.not_le <| hN x ⟨y, hne, hy⟩
  replace hN : ∀ (x), ∀ n ≥ N x + 1, ∀ y ∈ U x, f n y = f (N x + 1) y :=
    fun x n hn y hy => Nat.le_induction rfl (fun k hle => (hN x _ hle _ hy).trans) n hn
  /-
    X : Type u_4
    inst✝ : TopologicalSpace X
    π : X → Sort u_6
    f : Nat → (x : X) → π x
    U : X → Set X
    hUx : ∀ (x : X), Membership.mem (nhds x) (U x)
    hU : ∀ (x : X), (setOf fun i => (Inter.inter ((fun n => setOf fun x => Ne (f ( …
    N : X → Nat
    hN : ∀ (x : X) (n : Nat), GE.ge n (HAdd.hAdd (N x) 1) → ∀ (y : X), Membership. …
    ⊢ Exists fun F => ∀ (x : X), Filter.Eventually (fun p => Eq (f p.1 p.2) (F p.2 …
  -/
  refine ⟨fun x => f (N x + 1) x, fun x => ?_⟩
  /-
    X : Type u_4
    inst✝ : TopologicalSpace X
    π : X → Sort u_6
    f : Nat → (x : X) → π x
    U : X → Set X
    hUx : ∀ (x : X), Membership.mem (nhds x) (U x)
    hU : ∀ (x : X), (setOf fun i => (Inter.inter ((fun n => setOf fun x => Ne (f ( …
    N : X → Nat
    hN : ∀ (x : X) (n : Nat), GE.ge n (HAdd.hAdd (N x) 1) → ∀ (y : X), Membership. …
    x : X
    ⊢ Filter.Eventually (fun p => Eq (f p.1 p.2) ((fun x => f (HAdd.hAdd (N x) 1)  …
  -/
  filter_upwards [Filter.prod_mem_prod (eventually_gt_atTop (N x)) (hUx x)]
  /-
    case h
    X : Type u_4
    inst✝ : TopologicalSpace X
    π : X → Sort u_6
    f : Nat → (x : X) → π x
    U : X → Set X
    hUx : ∀ (x : X), Membership.mem (nhds x) (U x)
    hU : ∀ (x : X), (setOf fun i => (Inter.inter ((fun n => setOf fun x => Ne (f ( …
    N : X → Nat
    hN : ∀ (x : X) (n : Nat), GE.ge n (HAdd.hAdd (N x) 1) → ∀ (y : X), Membership. …
    x : X
    ⊢ ∀ (a : Prod Nat X), Membership.mem (SProd.sprod (setOf fun x_1 => LT.lt (N x …
  -/
  rintro ⟨n, y⟩ ⟨hn : N x < n, hy : y ∈ U x⟩
  calc
    f n y = f (N x + 1) y := hN _ _ hn _ hy
    _ = f (max (N x + 1) (N y + 1)) y := (hN _ _ (le_max_left _ _) _ hy).symm
    _ = f (N y + 1) y := hN _ _ (le_max_right _ _) _ (mem_of_mem_nhds <| hUx y)


/-- Let `f : ℕ → Π a, β a` be a sequence of (dependent) functions on a topological space. Suppose
that the family of sets `s n = {x | f (n + 1) x ≠ f n x}` is locally finite. Then there exists a
function `F : Π a, β a` such that for any `x`, for sufficiently large values of `n`, we have
`f n y = F y` in a neighbourhood of `x`. -/
theorem exists_forall_eventually_atTop_eventually_eq' {π : X → Sort*} {f : ℕ → ∀ x : X, π x}
    (hf : LocallyFinite fun n => { x | f (n + 1) x ≠ f n x }) :
    ∃ F : ∀ x : X, π x, ∀ x, ∀ᶠ n : ℕ in atTop, ∀ᶠ y : X in 𝓝 x, f n y = F y :=
  hf.exists_forall_eventually_eq_prod.imp fun _F hF x => (hF x).curry


/-- Let `f : ℕ → α → β` be a sequence of functions on a topological space. Suppose
that the family of sets `s n = {x | f (n + 1) x ≠ f n x}` is locally finite. Then there exists a
function `F :  α → β` such that for any `x`, for sufficiently large values of `n`, we have
`f n =ᶠ[𝓝 x] F`. -/
theorem exists_forall_eventually_atTop_eventuallyEq {f : ℕ → X → α}
    (hf : LocallyFinite fun n => { x | f (n + 1) x ≠ f n x }) :
    ∃ F : X → α, ∀ x, ∀ᶠ n : ℕ in atTop, f n =ᶠ[𝓝 x] F :=
  hf.exists_forall_eventually_atTop_eventually_eq'


theorem preimage_continuous {g : Y → X} (hf : LocallyFinite f) (hg : Continuous g) :
    LocallyFinite (g ⁻¹' f ·) := fun x =>
  let ⟨s, hsx, hs⟩ := hf (g x)
  ⟨g ⁻¹' s, hg.continuousAt hsx, hs.subset fun _ ⟨y, hy⟩ => ⟨g y, hy⟩⟩


theorem prod_right (hf : LocallyFinite f) (g : ι → Set Y) : LocallyFinite (fun i ↦ f i ×ˢ g i) :=
  (hf.preimage_continuous continuous_fst).subset fun _ ↦ prod_subset_preimage_fst _ _


theorem prod_left {g : ι → Set Y} (hg : LocallyFinite g) (f : ι → Set X) :
    LocallyFinite (fun i ↦ f i ×ˢ g i) :=
  (hg.preimage_continuous continuous_snd).subset fun _ ↦ prod_subset_preimage_snd _ _


@[simp]
theorem Equiv.locallyFinite_comp_iff (e : ι' ≃ ι) : LocallyFinite (f ∘ e) ↔ LocallyFinite f :=
               /-
                 ι : Type u_1
                 ι' : Type u_2
                 X : Type u_4
                 inst✝ : TopologicalSpace X
                 f : ι → Set X
                 e : Equiv ι' ι
                 h : LocallyFinite (Function.comp f ⇑e)
                 ⊢ LocallyFinite f
               -/
  ⟨fun h => by simpa only [comp_def, e.apply_symm_apply] using h.comp_injective e.symm.injective,
               /-
                 🎉 no goals
               -/
    fun h => h.comp_injective e.injective⟩


theorem locallyFinite_sum {f : ι ⊕ ι' → Set X} :
    LocallyFinite f ↔ LocallyFinite (f ∘ Sum.inl) ∧ LocallyFinite (f ∘ Sum.inr) := by
  simp only [locallyFinite_iff_smallSets, ← forall_and, ← finite_preimage_inl_and_inr,
    preimage_setOf_eq, (· ∘ ·), eventually_and]


theorem LocallyFinite.sum_elim {g : ι' → Set X} (hf : LocallyFinite f) (hg : LocallyFinite g) :
    LocallyFinite (Sum.elim f g) :=
  locallyFinite_sum.mpr ⟨hf, hg⟩


theorem locallyFinite_option {f : Option ι → Set X} :
    LocallyFinite f ↔ LocallyFinite (f ∘ some) := by
  /-
    ι : Type u_1
    X : Type u_4
    inst✝ : TopologicalSpace X
    f : Option ι → Set X
    ⊢ Iff (LocallyFinite f) (LocallyFinite (Function.comp f Option.some))
  -/
  rw [← (Equiv.optionEquivSumPUnit.{0, _} ι).symm.locallyFinite_comp_iff, locallyFinite_sum]
  /-
    ι : Type u_1
    X : Type u_4
    inst✝ : TopologicalSpace X
    f : Option ι → Set X
    ⊢ Iff (And (LocallyFinite (Function.comp (Function.comp f ⇑(Equiv.optionEquivS …
  -/
  simp only [locallyFinite_of_finite, and_true]
  /-
    ι : Type u_1
    X : Type u_4
    inst✝ : TopologicalSpace X
    f : Option ι → Set X
    ⊢ Iff (LocallyFinite (Function.comp (Function.comp f ⇑(Equiv.optionEquivSumPUn …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem LocallyFinite.option_elim' (hf : LocallyFinite f) (s : Set X) :
    LocallyFinite (Option.elim' s f) :=
  locallyFinite_option.2 hf


theorem LocallyFinite.eventually_subset {s : ι → Set X}
    (hs : LocallyFinite s) (hs' : ∀ i, IsClosed (s i)) (x : X) :
    ∀ᶠ y in 𝓝 x, {i | y ∈ s i} ⊆ {i | x ∈ s i} := by
  /-
    ι : Type u_1
    X : Type u_4
    inst✝ : TopologicalSpace X
    s : ι → Set X
    hs : LocallyFinite s
    hs' : ∀ (i : ι), IsClosed (s i)
    x : X
    ⊢ Filter.Eventually (fun y => HasSubset.Subset (setOf fun i => Membership.mem  …
  -/
  filter_upwards [hs.iInter_compl_mem_nhds hs' x] with y hy i hi
  /-
    case h
    ι : Type u_1
    X : Type u_4
    inst✝ : TopologicalSpace X
    s : ι → Set X
    hs : LocallyFinite s
    hs' : ∀ (i : ι), IsClosed (s i)
    x y : X
    hy : Membership.mem (Set.iInter fun i => Set.iInter fun x => HasCompl.compl (s …
    i : ι
    hi : Membership.mem (setOf fun i => Membership.mem (s i) y) i
    ⊢ Membership.mem (setOf fun i => Membership.mem (s i) x) i
  -/
  simp only [mem_iInter, mem_compl_iff] at hy
  /-
    case h
    ι : Type u_1
    X : Type u_4
    inst✝ : TopologicalSpace X
    s : ι → Set X
    hs : LocallyFinite s
    hs' : ∀ (i : ι), IsClosed (s i)
    x y : X
    i : ι
    hi : Membership.mem (setOf fun i => Membership.mem (s i) y) i
    hy : ∀ (i : ι), Not (Membership.mem (s i) x) → Not (Membership.mem (s i) y)
    ⊢ Membership.mem (setOf fun i => Membership.mem (s i) x) i
  -/
  exact not_imp_not.mp (hy i) hi
  /-
    🎉 no goals
  -/


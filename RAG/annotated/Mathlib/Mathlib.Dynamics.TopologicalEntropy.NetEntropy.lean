/-- Given a subset `F`, an entourage `U` and an integer `n`, a subset `s` of `F` is a
`(U, n)`-dynamical net of `F` if no two orbits of length `n` of points in `s` shadow each other.-/
def IsDynNetIn (T : X → X) (F : Set X) (U : Set (X × X)) (n : ℕ) (s : Set X) : Prop :=
  s ⊆ F ∧ s.PairwiseDisjoint (fun x : X ↦ ball x (dynEntourage T U n))


lemma IsDynNetIn.of_le {T : X → X} {F : Set X} {U : Set (X × X)} {m n : ℕ} (m_n : m ≤ n) {s : Set X}
    (h : IsDynNetIn T F U m s) :
    IsDynNetIn T F U n s :=
  ⟨h.1, PairwiseDisjoint.mono h.2 (fun x ↦ ball_mono (dynEntourage_antitone T U m_n) x)⟩


lemma IsDynNetIn.of_entourage_subset {T : X → X} {F : Set X} {U V : Set (X × X)} (U_V : U ⊆ V)
    {n : ℕ} {s : Set X} (h : IsDynNetIn T F V n s) :
    IsDynNetIn T F U n s :=
  ⟨h.1, PairwiseDisjoint.mono h.2 (fun x ↦ ball_mono (dynEntourage_monotone T n U_V) x)⟩


lemma isDynNetIn_empty {T : X → X} {F : Set X} {U : Set (X × X)} {n : ℕ} :
    IsDynNetIn T F U n ∅ :=
  ⟨empty_subset F, pairwise_empty _⟩


lemma isDynNetIn_singleton (T : X → X) {F : Set X} (U : Set (X × X)) (n : ℕ) {x : X} (h : x ∈ F) :
    IsDynNetIn T F U n {x} :=
  ⟨singleton_subset_iff.2 h, pairwise_singleton x _⟩


/-- Given an entourage `U` and a time `n`, a dynamical net has a smaller cardinality than
  a dynamical cover. This lemma is the first of two key results to compare two versions of
  topological entropy: with cover and with nets, the second being `coverMincard_le_netMaxcard`.-/
lemma IsDynNetIn.card_le_card_of_isDynCoverOf {T : X → X} {F : Set X} {U : Set (X × X)}
    (U_symm : SymmetricRel U) {n : ℕ} {s t : Finset X} (hs : IsDynNetIn T F U n s)
    (ht : IsDynCoverOf T F U n t) :
    s.card ≤ t.card := by
  have (x : X) (x_s : x ∈ s) : ∃ z ∈ t, x ∈ ball z (dynEntourage T U n) := by
    specialize ht (hs.1 x_s)
    simp only [Finset.coe_sort_coe, mem_iUnion, Subtype.exists, exists_prop] at ht
    exact ht
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    U_symm : SymmetricRel U
    n : Nat
    s t : Finset X
    hs : Dynamics.IsDynNetIn T F U n ↑s
    ht : Dynamics.IsDynCoverOf T F U n ↑t
    this : ∀ (x : X), Membership.mem s x → Exists fun z => And (Membership.mem t z …
    ⊢ LE.le s.card t.card
  -/
  choose! F s_t using this
  /-
    X : Type u_1
    T : X → X
    F✝ : Set X
    U : Set (Prod X X)
    U_symm : SymmetricRel U
    n : Nat
    s t : Finset X
    hs : Dynamics.IsDynNetIn T F✝ U n ↑s
    ht : Dynamics.IsDynCoverOf T F✝ U n ↑t
    F : X → X
    s_t : ∀ (x : X), Membership.mem s x → And (Membership.mem t (F x)) (Membership …
    ⊢ LE.le s.card t.card
  -/
  simp only [mem_ball_symmetry (U_symm.dynEntourage T n)] at s_t
  /-
    X : Type u_1
    T : X → X
    F✝ : Set X
    U : Set (Prod X X)
    U_symm : SymmetricRel U
    n : Nat
    s t : Finset X
    hs : Dynamics.IsDynNetIn T F✝ U n ↑s
    ht : Dynamics.IsDynCoverOf T F✝ U n ↑t
    F : X → X
    s_t : ∀ (x : X), Membership.mem s x → And (Membership.mem t (F x)) (Membership …
    ⊢ LE.le s.card t.card
  -/
  apply Finset.card_le_card_of_injOn F (fun x x_s ↦ (s_t x x_s).1)
  exact fun x x_s y y_s Fx_Fy ↦
    PairwiseDisjoint.elim_set hs.2 x_s y_s (F x) (s_t x x_s).2 (Fx_Fy ▸ (s_t y y_s).2)


/-- The largest cardinality of a `(U, n)`-dynamical net of `F`. Takes values in `ℕ∞`, and is
infinite if and only if `F` admits nets of arbitrarily large size.-/
noncomputable def netMaxcard (T : X → X) (F : Set X) (U : Set (X × X)) (n : ℕ) : ℕ∞ :=
  ⨆ (s : Finset X) (_ : IsDynNetIn T F U n s), (s.card : ℕ∞)


lemma IsDynNetIn.card_le_netMaxcard {T : X → X} {F : Set X} {U : Set (X × X)} {n : ℕ} {s : Finset X}
    (h : IsDynNetIn T F U n s) :
    s.card ≤ netMaxcard T F U n :=
  le_iSup₂ (α := ℕ∞) s h


lemma netMaxcard_monotone_time (T : X → X) (F : Set X) (U : Set (X × X)) :
    Monotone (fun n : ℕ ↦ netMaxcard T F U n) :=
  fun _ _ m_n ↦ biSup_mono (fun _ h ↦ h.of_le m_n)


lemma netMaxcard_antitone (T : X → X) (F : Set X) (n : ℕ) :
    Antitone (fun U : Set (X × X) ↦ netMaxcard T F U n) :=
  fun _ _ U_V ↦ biSup_mono (fun _ h ↦ h.of_entourage_subset U_V)


lemma netMaxcard_finite_iff (T : X → X) (F : Set X) (U : Set (X × X)) (n : ℕ) :
    netMaxcard T F U n < ⊤ ↔
    ∃ s : Finset X, IsDynNetIn T F U n s ∧ (s.card : ℕ∞) = netMaxcard T F U n := by
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    n : Nat
    ⊢ Iff (LT.lt (Dynamics.netMaxcard T F U n) Top.top) (Exists fun s => And (Dyna …
  -/
  apply Iff.intro <;> intro h
    /-
      case mp
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n : Nat
      h : LT.lt (Dynamics.netMaxcard T F U n) Top.top
      ⊢ Exists fun s => And (Dynamics.IsDynNetIn T F U n ↑s) (Eq (↑s.card) (Dynamics …
    -/
  · rcases WithTop.ne_top_iff_exists.1 h.ne with ⟨k, k_max⟩
    /-
      case mp.intro
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n : Nat
      h : LT.lt (Dynamics.netMaxcard T F U n) Top.top
      k : Nat
      k_max : Eq (↑k) (Dynamics.netMaxcard T F U n)
      ⊢ Exists fun s => And (Dynamics.IsDynNetIn T F U n ↑s) (Eq (↑s.card) (Dynamics …
    -/
    rw [← k_max]
    /-
      case mp.intro
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n : Nat
      h : LT.lt (Dynamics.netMaxcard T F U n) Top.top
      k : Nat
      k_max : Eq (↑k) (Dynamics.netMaxcard T F U n)
      ⊢ Exists fun s => And (Dynamics.IsDynNetIn T F U n ↑s) (Eq ↑s.card ↑k)
    -/
    simp only [ENat.some_eq_coe, Nat.cast_inj]
    -- The criterion we want to use is `Nat.sSup_mem`. We rewrite `netMaxcard` with an `sSup`,
    -- then check its `BddAbove` and `Nonempty` hypotheses.
    have : netMaxcard T F U n
      = sSup (WithTop.some '' (Finset.card '' {s : Finset X | IsDynNetIn T F U n s})) := by
      rw [netMaxcard, ← image_comp, sSup_image]
      simp only [mem_setOf_eq, ENat.some_eq_coe, Function.comp_apply]
    /-
      case mp.intro
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n : Nat
      h : LT.lt (Dynamics.netMaxcard T F U n) Top.top
      k : Nat
      k_max : Eq (↑k) (Dynamics.netMaxcard T F U n)
      this : Eq (Dynamics.netMaxcard T F U n) (SupSet.sSup (Set.image WithTop.some ( …
      ⊢ Exists fun s => And (Dynamics.IsDynNetIn T F U n ↑s) (Eq s.card k)
    -/
    rw [this] at k_max
    have h_bdda : BddAbove (Finset.card '' {s : Finset X | IsDynNetIn T F U n s}) := by
      refine ⟨k, mem_upperBounds.2 ?_⟩
      simp only [mem_image, mem_setOf_eq, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂]
      intro s h
      rw [← WithTop.coe_le_coe, k_max]
      apply le_sSup
      simp only [ENat.some_eq_coe, mem_image, mem_setOf_eq, Nat.cast_inj, exists_eq_right]
      exact Filter.frequently_principal.mp fun a ↦ a h rfl
    have h_nemp : (Finset.card '' {s : Finset X | IsDynNetIn T F U n s}).Nonempty := by
      refine ⟨0, ?_⟩
      simp only [mem_image, mem_setOf_eq, Finset.card_eq_zero, exists_eq_right, Finset.coe_empty]
      exact isDynNetIn_empty
    /-
      case mp.intro
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n : Nat
      h : LT.lt (Dynamics.netMaxcard T F U n) Top.top
      k : Nat
      k_max : Eq (↑k) (SupSet.sSup (Set.image WithTop.some (Set.image Finset.card (s …
      this : Eq (Dynamics.netMaxcard T F U n) (SupSet.sSup (Set.image WithTop.some ( …
      h_bdda : BddAbove (Set.image Finset.card (setOf fun s => Dynamics.IsDynNetIn T …
      h_nemp : (Set.image Finset.card (setOf fun s => Dynamics.IsDynNetIn T F U n ↑s …
      ⊢ Exists fun s => And (Dynamics.IsDynNetIn T F U n ↑s) (Eq s.card k)
    -/
    rw [← WithTop.coe_sSup' h_bdda, ENat.some_eq_coe, Nat.cast_inj] at k_max
    /-
      case mp.intro
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n : Nat
      h : LT.lt (Dynamics.netMaxcard T F U n) Top.top
      k : Nat
      k_max : Eq k (SupSet.sSup (Set.image Finset.card (setOf fun s => Dynamics.IsDy …
      this : Eq (Dynamics.netMaxcard T F U n) (SupSet.sSup (Set.image WithTop.some ( …
      h_bdda : BddAbove (Set.image Finset.card (setOf fun s => Dynamics.IsDynNetIn T …
      h_nemp : (Set.image Finset.card (setOf fun s => Dynamics.IsDynNetIn T F U n ↑s …
      ⊢ Exists fun s => And (Dynamics.IsDynNetIn T F U n ↑s) (Eq s.card k)
    -/
    have key := Nat.sSup_mem h_nemp h_bdda
    /-
      case mp.intro
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n : Nat
      h : LT.lt (Dynamics.netMaxcard T F U n) Top.top
      k : Nat
      k_max : Eq k (SupSet.sSup (Set.image Finset.card (setOf fun s => Dynamics.IsDy …
      this : Eq (Dynamics.netMaxcard T F U n) (SupSet.sSup (Set.image WithTop.some ( …
      h_bdda : BddAbove (Set.image Finset.card (setOf fun s => Dynamics.IsDynNetIn T …
      h_nemp : (Set.image Finset.card (setOf fun s => Dynamics.IsDynNetIn T F U n ↑s …
      key : Membership.mem (Set.image Finset.card (setOf fun s => Dynamics.IsDynNetI …
      ⊢ Exists fun s => And (Dynamics.IsDynNetIn T F U n ↑s) (Eq s.card k)
    -/
    rw [← k_max, mem_image] at key
    /-
      case mp.intro
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n : Nat
      h : LT.lt (Dynamics.netMaxcard T F U n) Top.top
      k : Nat
      k_max : Eq k (SupSet.sSup (Set.image Finset.card (setOf fun s => Dynamics.IsDy …
      this : Eq (Dynamics.netMaxcard T F U n) (SupSet.sSup (Set.image WithTop.some ( …
      h_bdda : BddAbove (Set.image Finset.card (setOf fun s => Dynamics.IsDynNetIn T …
      h_nemp : (Set.image Finset.card (setOf fun s => Dynamics.IsDynNetIn T F U n ↑s …
      key : Exists fun x => And (Membership.mem (setOf fun s => Dynamics.IsDynNetIn  …
      ⊢ Exists fun s => And (Dynamics.IsDynNetIn T F U n ↑s) (Eq s.card k)
    -/
    simp only [mem_setOf_eq] at key
    /-
      case mp.intro
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n : Nat
      h : LT.lt (Dynamics.netMaxcard T F U n) Top.top
      k : Nat
      k_max : Eq k (SupSet.sSup (Set.image Finset.card (setOf fun s => Dynamics.IsDy …
      this : Eq (Dynamics.netMaxcard T F U n) (SupSet.sSup (Set.image WithTop.some ( …
      h_bdda : BddAbove (Set.image Finset.card (setOf fun s => Dynamics.IsDynNetIn T …
      h_nemp : (Set.image Finset.card (setOf fun s => Dynamics.IsDynNetIn T F U n ↑s …
      key : Exists fun x => And (Dynamics.IsDynNetIn T F U n ↑x) (Eq x.card k)
      ⊢ Exists fun s => And (Dynamics.IsDynNetIn T F U n ↑s) (Eq s.card k)
    -/
    exact key
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n : Nat
      h : Exists fun s => And (Dynamics.IsDynNetIn T F U n ↑s) (Eq (↑s.card) (Dynami …
      ⊢ LT.lt (Dynamics.netMaxcard T F U n) Top.top
    -/
  · rcases h with ⟨s, _, s_netMaxcard⟩
    /-
      case mpr.intro.intro
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n : Nat
      s : Finset X
      left✝ : Dynamics.IsDynNetIn T F U n ↑s
      s_netMaxcard : Eq (↑s.card) (Dynamics.netMaxcard T F U n)
      ⊢ LT.lt (Dynamics.netMaxcard T F U n) Top.top
    -/
    rw [← s_netMaxcard]
    /-
      case mpr.intro.intro
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n : Nat
      s : Finset X
      left✝ : Dynamics.IsDynNetIn T F U n ↑s
      s_netMaxcard : Eq (↑s.card) (Dynamics.netMaxcard T F U n)
      ⊢ LT.lt (↑s.card) Top.top
    -/
    exact WithTop.coe_lt_top s.card
    /-
      🎉 no goals
    -/


@[simp]
lemma netMaxcard_empty {T : X → X} {U : Set (X × X)} {n : ℕ} : netMaxcard T ∅ U n = 0 := by
  /-
    X : Type u_1
    T : X → X
    U : Set (Prod X X)
    n : Nat
    ⊢ Eq (Dynamics.netMaxcard T EmptyCollection.emptyCollection U n) 0
  -/
  rw [netMaxcard, ← bot_eq_zero, iSup₂_eq_bot]
  /-
    X : Type u_1
    T : X → X
    U : Set (Prod X X)
    n : Nat
    ⊢ ∀ (i : Finset X), Dynamics.IsDynNetIn T EmptyCollection.emptyCollection U n  …
  -/
  intro s s_net
  /-
    X : Type u_1
    T : X → X
    U : Set (Prod X X)
    n : Nat
    s : Finset X
    s_net : Dynamics.IsDynNetIn T EmptyCollection.emptyCollection U n ↑s
    ⊢ Eq (↑s.card) Bot.bot
  -/
  replace s_net := subset_empty_iff.1 s_net.1
  /-
    X : Type u_1
    T : X → X
    U : Set (Prod X X)
    n : Nat
    s : Finset X
    s_net : Eq (↑s) EmptyCollection.emptyCollection
    ⊢ Eq (↑s.card) Bot.bot
  -/
  norm_cast at s_net
  /-
    X : Type u_1
    T : X → X
    U : Set (Prod X X)
    n : Nat
    s : Finset X
    s_net : Eq s EmptyCollection.emptyCollection
    ⊢ Eq (↑s.card) Bot.bot
  -/
  rw [s_net, Finset.card_empty, CharP.cast_eq_zero, bot_eq_zero']
  /-
    🎉 no goals
  -/


lemma netMaxcard_eq_zero_iff (T : X → X) (F : Set X) (U : Set (X × X)) (n : ℕ) :
    netMaxcard T F U n = 0 ↔ F = ∅ := by
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    n : Nat
    ⊢ Iff (Eq (Dynamics.netMaxcard T F U n) 0) (Eq F EmptyCollection.emptyCollecti …
  -/
  refine Iff.intro (fun h ↦ ?_) (fun h ↦ by rw [h, netMaxcard_empty])
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    n : Nat
    h : Eq (Dynamics.netMaxcard T F U n) 0
    ⊢ Eq F EmptyCollection.emptyCollection
  -/
  rw [eq_empty_iff_forall_not_mem]
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    n : Nat
    h : Eq (Dynamics.netMaxcard T F U n) 0
    ⊢ ∀ (x : X), Not (Membership.mem F x)
  -/
  intro x x_F
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    n : Nat
    h : Eq (Dynamics.netMaxcard T F U n) 0
    x : X
    x_F : Membership.mem F x
    ⊢ False
  -/
  have key := isDynNetIn_singleton T U n x_F
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    n : Nat
    h : Eq (Dynamics.netMaxcard T F U n) 0
    x : X
    x_F : Membership.mem F x
    key : Dynamics.IsDynNetIn T F U n (Singleton.singleton x)
    ⊢ False
  -/
  rw [← Finset.coe_singleton] at key
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    n : Nat
    h : Eq (Dynamics.netMaxcard T F U n) 0
    x : X
    x_F : Membership.mem F x
    key : Dynamics.IsDynNetIn T F U n ↑(Singleton.singleton x)
    ⊢ False
  -/
  replace key := key.card_le_netMaxcard
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    n : Nat
    h : Eq (Dynamics.netMaxcard T F U n) 0
    x : X
    x_F : Membership.mem F x
    key : LE.le (↑(Singleton.singleton x).card) (Dynamics.netMaxcard T F U n)
    ⊢ False
  -/
  rw [Finset.card_singleton, Nat.cast_one, h] at key
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    n : Nat
    h : Eq (Dynamics.netMaxcard T F U n) 0
    x : X
    x_F : Membership.mem F x
    key : LE.le 1 0
    ⊢ False
  -/
  exact key.not_lt zero_lt_one
  /-
    🎉 no goals
  -/


lemma one_le_netMaxcard_iff (T : X → X) (F : Set X) (U : Set (X × X)) (n : ℕ) :
    1 ≤ netMaxcard T F U n ↔ F.Nonempty := by
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    n : Nat
    ⊢ Iff (LE.le 1 (Dynamics.netMaxcard T F U n)) F.Nonempty
  -/
  rw [ENat.one_le_iff_ne_zero, nonempty_iff_ne_empty]
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    n : Nat
    ⊢ Iff (Ne (Dynamics.netMaxcard T F U n) 0) (Ne F EmptyCollection.emptyCollecti …
  -/
  exact not_iff_not.2 (netMaxcard_eq_zero_iff T F U n)
  /-
    🎉 no goals
  -/


lemma netMaxcard_zero (T : X → X) {F : Set X} (h : F.Nonempty) (U : Set (X × X)) :
    netMaxcard T F U 0 = 1 := by
  /-
    X : Type u_1
    T : X → X
    F : Set X
    h : F.Nonempty
    U : Set (Prod X X)
    ⊢ Eq (Dynamics.netMaxcard T F U 0) 1
  -/
  apply (iSup₂_le _).antisymm ((one_le_netMaxcard_iff T F U 0).2 h)
  /-
    X : Type u_1
    T : X → X
    F : Set X
    h : F.Nonempty
    U : Set (Prod X X)
    ⊢ ∀ (i : Finset X), Dynamics.IsDynNetIn T F U 0 ↑i → LE.le (↑i.card) 1
  -/
  intro s ⟨_, s_net⟩
  /-
    X : Type u_1
    T : X → X
    F : Set X
    h : F.Nonempty
    U : Set (Prod X X)
    s : Finset X
    left✝ : HasSubset.Subset (↑s) F
    s_net : (↑s).PairwiseDisjoint fun x => UniformSpace.ball x (Dynamics.dynEntour …
    ⊢ LE.le (↑s.card) 1
  -/
  simp only [ball, dynEntourage_zero, preimage_univ] at s_net
  /-
    X : Type u_1
    T : X → X
    F : Set X
    h : F.Nonempty
    U : Set (Prod X X)
    s : Finset X
    left✝ : HasSubset.Subset (↑s) F
    s_net : (↑s).PairwiseDisjoint fun x => Set.univ
    ⊢ LE.le (↑s.card) 1
  -/
  norm_cast
  /-
    X : Type u_1
    T : X → X
    F : Set X
    h : F.Nonempty
    U : Set (Prod X X)
    s : Finset X
    left✝ : HasSubset.Subset (↑s) F
    s_net : (↑s).PairwiseDisjoint fun x => Set.univ
    ⊢ LE.le s.card 1
  -/
  refine Finset.card_le_one.2 (fun x x_s y y_s ↦ ?_)
  /-
    X : Type u_1
    T : X → X
    F : Set X
    h : F.Nonempty
    U : Set (Prod X X)
    s : Finset X
    left✝ : HasSubset.Subset (↑s) F
    s_net : (↑s).PairwiseDisjoint fun x => Set.univ
    x : X
    x_s : Membership.mem s x
    y : X
    y_s : Membership.mem s y
    ⊢ Eq x y
  -/
  exact PairwiseDisjoint.elim_set s_net x_s y_s x (mem_univ x) (mem_univ x)
  /-
    🎉 no goals
  -/


lemma netMaxcard_univ (T : X → X) {F : Set X} (h : F.Nonempty) (n : ℕ) :
    netMaxcard T F univ n = 1 := by
  /-
    X : Type u_1
    T : X → X
    F : Set X
    h : F.Nonempty
    n : Nat
    ⊢ Eq (Dynamics.netMaxcard T F Set.univ n) 1
  -/
  apply (iSup₂_le _).antisymm ((one_le_netMaxcard_iff T F univ n).2 h)
  /-
    X : Type u_1
    T : X → X
    F : Set X
    h : F.Nonempty
    n : Nat
    ⊢ ∀ (i : Finset X), Dynamics.IsDynNetIn T F Set.univ n ↑i → LE.le (↑i.card) 1
  -/
  intro s ⟨_, s_net⟩
  /-
    X : Type u_1
    T : X → X
    F : Set X
    h : F.Nonempty
    n : Nat
    s : Finset X
    left✝ : HasSubset.Subset (↑s) F
    s_net : (↑s).PairwiseDisjoint fun x => UniformSpace.ball x (Dynamics.dynEntour …
    ⊢ LE.le (↑s.card) 1
  -/
  simp only [ball, dynEntourage_univ, preimage_univ] at s_net
  /-
    X : Type u_1
    T : X → X
    F : Set X
    h : F.Nonempty
    n : Nat
    s : Finset X
    left✝ : HasSubset.Subset (↑s) F
    s_net : (↑s).PairwiseDisjoint fun x => Set.univ
    ⊢ LE.le (↑s.card) 1
  -/
  norm_cast
  /-
    X : Type u_1
    T : X → X
    F : Set X
    h : F.Nonempty
    n : Nat
    s : Finset X
    left✝ : HasSubset.Subset (↑s) F
    s_net : (↑s).PairwiseDisjoint fun x => Set.univ
    ⊢ LE.le s.card 1
  -/
  refine Finset.card_le_one.2 (fun x x_s y y_s ↦ ?_)
  /-
    X : Type u_1
    T : X → X
    F : Set X
    h : F.Nonempty
    n : Nat
    s : Finset X
    left✝ : HasSubset.Subset (↑s) F
    s_net : (↑s).PairwiseDisjoint fun x => Set.univ
    x : X
    x_s : Membership.mem s x
    y : X
    y_s : Membership.mem s y
    ⊢ Eq x y
  -/
  exact PairwiseDisjoint.elim_set s_net x_s y_s x (mem_univ x) (mem_univ x)
  /-
    🎉 no goals
  -/


lemma netMaxcard_infinite_iff (T : X → X) (F : Set X) (U : Set (X × X)) (n : ℕ) :
    netMaxcard T F U n = ⊤ ↔ ∀ k : ℕ, ∃ s : Finset X, IsDynNetIn T F U n s ∧ k ≤ s.card := by
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    n : Nat
    ⊢ Iff (Eq (Dynamics.netMaxcard T F U n) Top.top) (∀ (k : Nat), Exists fun s => …
  -/
  apply Iff.intro <;> intro h
    /-
      case mp
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n : Nat
      h : Eq (Dynamics.netMaxcard T F U n) Top.top
      ⊢ ∀ (k : Nat), Exists fun s => And (Dynamics.IsDynNetIn T F U n ↑s) (LE.le k s …
    -/
  · intro k
    /-
      case mp
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n : Nat
      h : Eq (Dynamics.netMaxcard T F U n) Top.top
      k : Nat
      ⊢ Exists fun s => And (Dynamics.IsDynNetIn T F U n ↑s) (LE.le k s.card)
    -/
    rw [netMaxcard, iSup_subtype', iSup_eq_top] at h
    /-
      case mp
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n : Nat
      h : ∀ (b : ENat), LT.lt b Top.top → Exists fun i => LT.lt b ↑(↑i).card
      k : Nat
      ⊢ Exists fun s => And (Dynamics.IsDynNetIn T F U n ↑s) (LE.le k s.card)
    -/
    specialize h k (ENat.coe_lt_top k)
    /-
      case mp
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n k : Nat
      h : Exists fun i => LT.lt ↑k ↑(↑i).card
      ⊢ Exists fun s => And (Dynamics.IsDynNetIn T F U n ↑s) (LE.le k s.card)
    -/
    simp only [Nat.cast_lt, Subtype.exists, exists_prop] at h
    /-
      case mp
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n k : Nat
      h : Exists fun a => And (Dynamics.IsDynNetIn T F U n ↑a) (LT.lt k a.card)
      ⊢ Exists fun s => And (Dynamics.IsDynNetIn T F U n ↑s) (LE.le k s.card)
    -/
    rcases h with ⟨s, s_net, s_k⟩
    /-
      case mp.intro.intro
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n k : Nat
      s : Finset X
      s_net : Dynamics.IsDynNetIn T F U n ↑s
      s_k : LT.lt k s.card
      ⊢ Exists fun s => And (Dynamics.IsDynNetIn T F U n ↑s) (LE.le k s.card)
    -/
    exact ⟨s, ⟨s_net, s_k.le⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n : Nat
      h : ∀ (k : Nat), Exists fun s => And (Dynamics.IsDynNetIn T F U n ↑s) (LE.le k …
      ⊢ Eq (Dynamics.netMaxcard T F U n) Top.top
    -/
  · refine WithTop.forall_gt_iff_eq_top.1 fun k ↦ ?_
    /-
      case mpr
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n : Nat
      h : ∀ (k : Nat), Exists fun s => And (Dynamics.IsDynNetIn T F U n ↑s) (LE.le k …
      k : Nat
      ⊢ LT.lt (↑k) (Dynamics.netMaxcard T F U n)
    -/
    specialize h (k + 1)
    /-
      case mpr
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n k : Nat
      h : Exists fun s => And (Dynamics.IsDynNetIn T F U n ↑s) (LE.le (HAdd.hAdd k 1 …
      ⊢ LT.lt (↑k) (Dynamics.netMaxcard T F U n)
    -/
    rcases h with ⟨s, s_net, s_card⟩
    /-
      case mpr.intro.intro
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n k : Nat
      s : Finset X
      s_net : Dynamics.IsDynNetIn T F U n ↑s
      s_card : LE.le (HAdd.hAdd k 1) s.card
      ⊢ LT.lt (↑k) (Dynamics.netMaxcard T F U n)
    -/
    apply s_net.card_le_netMaxcard.trans_lt'
    /-
      case mpr.intro.intro
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n k : Nat
      s : Finset X
      s_net : Dynamics.IsDynNetIn T F U n ↑s
      s_card : LE.le (HAdd.hAdd k 1) s.card
      ⊢ LT.lt ↑k ↑s.card
    -/
    rw [ENat.some_eq_coe, Nat.cast_lt]
    /-
      case mpr.intro.intro
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      n k : Nat
      s : Finset X
      s_net : Dynamics.IsDynNetIn T F U n ↑s
      s_card : LE.le (HAdd.hAdd k 1) s.card
      ⊢ LT.lt k s.card
    -/
    exact (lt_add_one k).trans_le s_card
    /-
      🎉 no goals
    -/


lemma netMaxcard_le_coverMincard (T : X → X) (F : Set X) {U : Set (X × X)} (U_symm : SymmetricRel U)
    (n : ℕ) :
    netMaxcard T F U n ≤ coverMincard T F U n := by
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    U_symm : SymmetricRel U
    n : Nat
    ⊢ LE.le (Dynamics.netMaxcard T F U n) (Dynamics.coverMincard T F U n)
  -/
  rcases eq_top_or_lt_top (coverMincard T F U n) with h | h
    /-
      case inl
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      U_symm : SymmetricRel U
      n : Nat
      h : Eq (Dynamics.coverMincard T F U n) Top.top
      ⊢ LE.le (Dynamics.netMaxcard T F U n) (Dynamics.coverMincard T F U n)
    -/
  · exact h ▸ le_top
    /-
      🎉 no goals
    -/
    /-
      case inr
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      U_symm : SymmetricRel U
      n : Nat
      h : LT.lt (Dynamics.coverMincard T F U n) Top.top
      ⊢ LE.le (Dynamics.netMaxcard T F U n) (Dynamics.coverMincard T F U n)
    -/
  · rcases ((coverMincard_finite_iff T F U n).1 h) with ⟨t, t_cover, t_mincard⟩
    /-
      case inr.intro.intro
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      U_symm : SymmetricRel U
      n : Nat
      h : LT.lt (Dynamics.coverMincard T F U n) Top.top
      t : Finset X
      t_cover : Dynamics.IsDynCoverOf T F U n ↑t
      t_mincard : Eq (↑t.card) (Dynamics.coverMincard T F U n)
      ⊢ LE.le (Dynamics.netMaxcard T F U n) (Dynamics.coverMincard T F U n)
    -/
    rw [← t_mincard]
    /-
      case inr.intro.intro
      X : Type u_1
      T : X → X
      F : Set X
      U : Set (Prod X X)
      U_symm : SymmetricRel U
      n : Nat
      h : LT.lt (Dynamics.coverMincard T F U n) Top.top
      t : Finset X
      t_cover : Dynamics.IsDynCoverOf T F U n ↑t
      t_mincard : Eq (↑t.card) (Dynamics.coverMincard T F U n)
      ⊢ LE.le (Dynamics.netMaxcard T F U n) ↑t.card
    -/
    exact iSup₂_le (fun s s_net ↦ Nat.cast_le.2 (s_net.card_le_card_of_isDynCoverOf U_symm t_cover))
    /-
      🎉 no goals
    -/


/-- Given an entourage `U` and a time `n`, a minimal dynamical cover by `U ○ U` has a smaller
  cardinality than a maximal dynamical net by `U`. This lemma is the second of two key results to
  compare two versions topological entropy: with cover and with nets.-/
lemma coverMincard_le_netMaxcard (T : X → X) (F : Set X) {U : Set (X × X)} (U_rfl : idRel ⊆ U)
    (U_symm : SymmetricRel U) (n : ℕ) :
    coverMincard T F (U ○ U) n ≤ netMaxcard T F U n := by
  classical
  -- WLOG, there exists a maximal dynamical net `s`.
  rcases (eq_top_or_lt_top (netMaxcard T F U n)) with h | h
  · exact h ▸ le_top
  rcases ((netMaxcard_finite_iff T F U n).1 h) with ⟨s, s_net, s_netMaxcard⟩
  rw [← s_netMaxcard]
  apply IsDynCoverOf.coverMincard_le_card
  --  We have to check that `s` is a cover for `dynEntourage T F (U ○ U) n`.
  -- If `s` is not a cover, then we can add to `s` a point `x` which is not covered
  -- and get a new net. This contradicts the maximality of `s`.
  by_contra h
  rcases not_subset.1 h with ⟨x, x_F, x_uncov⟩
  simp only [Finset.mem_coe, mem_iUnion, exists_prop, not_exists, not_and] at x_uncov
  have larger_net : IsDynNetIn T F U n (insert x s) :=
    And.intro (insert_subset x_F s_net.1) (pairwiseDisjoint_insert.2 (And.intro s_net.2
      (fun y y_s _ ↦ (disjoint_left.2 (fun z z_x z_y ↦ x_uncov y y_s
        (mem_ball_dynEntourage_comp T n U_symm x y (nonempty_of_mem ⟨z_x, z_y⟩)))))))
  rw [← Finset.coe_insert x s] at larger_net
  apply larger_net.card_le_netMaxcard.not_lt
  rw [← s_netMaxcard, Nat.cast_lt]
  refine (lt_add_one s.card).trans_eq (Finset.card_insert_of_not_mem fun x_s ↦ ?_).symm
  apply x_uncov x x_s (ball_mono (dynEntourage_monotone T n (subset_comp_self U_rfl)) x
    (ball_mono (idRel_subset_dynEntourage T U_rfl n) x _))
  simp only [ball, mem_preimage, mem_idRel]


lemma log_netMaxcard_nonneg (T : X → X) {F : Set X} (h : F.Nonempty) (U : Set (X × X)) (n : ℕ) :
    0 ≤ log (netMaxcard T F U n) := by
  /-
    X : Type u_1
    T : X → X
    F : Set X
    h : F.Nonempty
    U : Set (Prod X X)
    n : Nat
    ⊢ LE.le 0 (↑(Dynamics.netMaxcard T F U n)).log
  -/
  apply zero_le_log_iff.2
  /-
    X : Type u_1
    T : X → X
    F : Set X
    h : F.Nonempty
    U : Set (Prod X X)
    n : Nat
    ⊢ LE.le 1 ↑(Dynamics.netMaxcard T F U n)
  -/
  rw [← ENat.toENNReal_one, ENat.toENNReal_le]
  /-
    X : Type u_1
    T : X → X
    F : Set X
    h : F.Nonempty
    U : Set (Prod X X)
    n : Nat
    ⊢ LE.le 1 (Dynamics.netMaxcard T F U n)
  -/
  exact (one_le_netMaxcard_iff T F U n).2 h
  /-
    🎉 no goals
  -/


/-- The entropy of an entourage `U`, defined as the exponential rate of growth of the size of the
largest `(U, n)`-dynamical net of `F`. Takes values in the space of extended real numbers
`[-∞,+∞]`. This version uses a `limsup`, and is chosen as the default definition.-/
noncomputable def netEntropyEntourage (T : X → X) (F : Set X) (U : Set (X × X)) :=
  atTop.limsup fun n : ℕ ↦ log (netMaxcard T F U n) / n


/-- The entropy of an entourage `U`, defined as the exponential rate of growth of the size of the
largest `(U, n)`-dynamical net of `F`. Takes values in the space of extended real numbers
`[-∞,+∞]`. This version uses a `liminf`, and is an alternative definition.-/
noncomputable def netEntropyInfEntourage (T : X → X) (F : Set X) (U : Set (X × X)) :=
  atTop.liminf fun n : ℕ ↦ log (netMaxcard T F U n) / n


lemma netEntropyInfEntourage_antitone (T : X → X) (F : Set X) :
    Antitone (fun U : Set (X × X) ↦ netEntropyInfEntourage T F U) :=
                 /-
                   X : Type u_1
                   T : X → X
                   F : Set X
                   x✝¹ x✝ : Set (Prod X X)
                   U_V : LE.le x✝¹ x✝
                   h : Filter.Eventually (fun a => LE.le (HDiv.hDiv (↑(Dynamics.netMaxcard T F x✝ …
                   ⊢ Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) Filter.atTop fun n => HDiv. …
                 -/
                 /-
                   🎉 no goals
                 -/
  fun _ _ U_V ↦ (liminf_le_liminf) (Eventually.of_forall
                 /-
                   🎉 no goals
                 -/
    fun n ↦ monotone_div_right_of_nonneg (Nat.cast_nonneg' n)
      (log_monotone (ENat.toENNReal_mono (netMaxcard_antitone T F n U_V))))


lemma netEntropyEntourage_antitone (T : X → X) (F : Set X) :
    Antitone (fun U : Set (X × X) ↦ netEntropyEntourage T F U) :=
                 /-
                   X : Type u_1
                   T : X → X
                   F : Set X
                   x✝¹ x✝ : Set (Prod X X)
                   U_V : LE.le x✝¹ x✝
                   h : Filter.atTop.EventuallyLE (fun n => HDiv.hDiv (↑(Dynamics.netMaxcard T F x …
                   ⊢ Filter.IsCoboundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun n => HDi …
                 -/
                 /-
                   🎉 no goals
                 -/
  fun _ _ U_V ↦ (limsup_le_limsup) (Eventually.of_forall
                 /-
                   🎉 no goals
                 -/
    fun n ↦ (monotone_div_right_of_nonneg (Nat.cast_nonneg' n)
      (log_monotone (ENat.toENNReal_mono (netMaxcard_antitone T F n U_V)))))


lemma netEntropyInfEntourage_le_netEntropyEntourage (T : X → X) (F : Set X) (U : Set (X × X)) :
                                                                /-
                                                                  X : Type u_1
                                                                  T : X → X
                                                                  F : Set X
                                                                  U : Set (Prod X X)
                                                                  ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun n => HDiv. …
                                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
    netEntropyInfEntourage T F U ≤ netEntropyEntourage T F U := liminf_le_limsup
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
lemma netEntropyEntourage_empty {T : X → X} {U : Set (X × X)} : netEntropyEntourage T ∅ U = ⊥ := by
  suffices h : ∀ᶠ n : ℕ in atTop, log (netMaxcard T ∅ U n) / n = ⊥ by
    rw [netEntropyEntourage, limsup_congr h]
    exact limsup_const ⊥
  /-
    X : Type u_1
    T : X → X
    U : Set (Prod X X)
    ⊢ Filter.Eventually (fun n => Eq (HDiv.hDiv (↑(Dynamics.netMaxcard T EmptyColl …
  -/
  simp only [netMaxcard_empty, ENat.toENNReal_zero, log_zero, eventually_atTop]
  /-
    X : Type u_1
    T : X → X
    U : Set (Prod X X)
    ⊢ Exists fun a => ∀ (b : Nat), GE.ge b a → Eq (HDiv.hDiv Bot.bot ↑b) Bot.bot
  -/
  exact ⟨1, fun n n_pos ↦ bot_div_of_pos_ne_top (Nat.cast_pos'.2 n_pos) (natCast_ne_top n)⟩
  /-
    🎉 no goals
  -/


@[simp]
lemma netEntropyInfEntourage_empty {T : X → X} {U : Set (X × X)} :
    netEntropyInfEntourage T ∅ U = ⊥ :=
  eq_bot_mono (netEntropyInfEntourage_le_netEntropyEntourage T ∅ U) netEntropyEntourage_empty


lemma netEntropyInfEntourage_nonneg (T : X → X) {F : Set X} (h : F.Nonempty) (U : Set (X × X)) :
    0 ≤ netEntropyInfEntourage T F U :=
  (le_iInf fun n ↦ div_nonneg (log_netMaxcard_nonneg T h U n) (Nat.cast_nonneg' n)).trans
    iInf_le_liminf


lemma netEntropyEntourage_nonneg (T : X → X) {F : Set X} (h : F.Nonempty) (U : Set (X × X)) :
    0 ≤ netEntropyEntourage T F U :=
  (netEntropyInfEntourage_nonneg T h U).trans (netEntropyInfEntourage_le_netEntropyEntourage T F U)


lemma netEntropyInfEntourage_univ (T : X → X) {F : Set X} (h : F.Nonempty) :
                                              /-
                                                X : Type u_1
                                                T : X → X
                                                F : Set X
                                                h : F.Nonempty
                                                ⊢ Eq (Dynamics.netEntropyInfEntourage T F Set.univ) 0
                                              -/
    netEntropyInfEntourage T F univ = 0 := by simp [netEntropyInfEntourage, netMaxcard_univ T h]
                                              /-
                                                🎉 no goals
                                              -/


lemma netEntropyEntourage_univ (T : X → X) {F : Set X} (h : F.Nonempty) :
                                           /-
                                             X : Type u_1
                                             T : X → X
                                             F : Set X
                                             h : F.Nonempty
                                             ⊢ Eq (Dynamics.netEntropyEntourage T F Set.univ) 0
                                           -/
    netEntropyEntourage T F univ = 0 := by simp [netEntropyEntourage, netMaxcard_univ T h]
                                           /-
                                             🎉 no goals
                                           -/


lemma netEntropyInfEntourage_le_coverEntropyInfEntourage (T : X → X) (F : Set X) {U : Set (X × X)}
    (U_symm : SymmetricRel U) :
    netEntropyInfEntourage T F U ≤ coverEntropyInfEntourage T F U :=
   /-
     X : Type u_1
     T : X → X
     F : Set X
     U : Set (Prod X X)
     U_symm : SymmetricRel U
     h : Filter.Eventually (fun a => LE.le (HDiv.hDiv (↑(Dynamics.netMaxcard T F U  …
     ⊢ Filter.IsBoundedUnder (fun x1 x2 => GE.ge x1 x2) Filter.atTop fun n => HDiv. …
   -/
   /-
     🎉 no goals
   -/
  (liminf_le_liminf) (Eventually.of_forall fun n ↦ (div_le_div_right_of_nonneg (Nat.cast_nonneg' n)
   /-
     🎉 no goals
   -/
    (log_monotone (ENat.toENNReal_le.2 (netMaxcard_le_coverMincard T F U_symm n)))))


lemma coverEntropyInfEntourage_le_netEntropyInfEntourage (T : X → X) (F : Set X) {U : Set (X × X)}
    (U_rfl : idRel ⊆ U) (U_symm : SymmetricRel U) :
    coverEntropyInfEntourage T F (U ○ U) ≤ netEntropyInfEntourage T F U := by
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    U_rfl : HasSubset.Subset idRel U
    U_symm : SymmetricRel U
    ⊢ LE.le (Dynamics.coverEntropyInfEntourage T F (compRel U U)) (Dynamics.netEnt …
  -/
  refine (liminf_le_liminf) (Eventually.of_forall fun n ↦ ?_)
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    U_rfl : HasSubset.Subset idRel U
    U_symm : SymmetricRel U
    n : Nat
    ⊢ LE.le (HDiv.hDiv (↑(Dynamics.coverMincard T F (compRel U U) n)).log ↑n) (HDi …
  -/
  apply div_le_div_right_of_nonneg (Nat.cast_nonneg' n) (log_monotone _)
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    U_rfl : HasSubset.Subset idRel U
    U_symm : SymmetricRel U
    n : Nat
    ⊢ LE.le ↑(Dynamics.coverMincard T F (compRel U U) n) ↑(Dynamics.netMaxcard T F …
  -/
  exact ENat.toENNReal_le.2 (coverMincard_le_netMaxcard T F U_rfl U_symm n)
  /-
    🎉 no goals
  -/


lemma netEntropyEntourage_le_coverEntropyEntourage (T : X → X) (F : Set X) {U : Set (X × X)}
    (U_symm : SymmetricRel U) :
    netEntropyEntourage T F U ≤ coverEntropyEntourage T F U := by
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    U_symm : SymmetricRel U
    ⊢ LE.le (Dynamics.netEntropyEntourage T F U) (Dynamics.coverEntropyEntourage T …
  -/
  refine (limsup_le_limsup) (Eventually.of_forall fun n ↦ ?_)
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    U_symm : SymmetricRel U
    n : Nat
    ⊢ LE.le ((fun n => HDiv.hDiv (↑(Dynamics.netMaxcard T F U n)).log ↑n) n) ((fun …
  -/
  apply div_le_div_right_of_nonneg (Nat.cast_nonneg' n) (log_monotone _)
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    U_symm : SymmetricRel U
    n : Nat
    ⊢ LE.le ↑(Dynamics.netMaxcard T F U n) ↑(Dynamics.coverMincard T F U n)
  -/
  exact ENat.toENNReal_le.2 (netMaxcard_le_coverMincard T F U_symm n)
  /-
    🎉 no goals
  -/


lemma coverEntropyEntourage_le_netEntropyEntourage (T : X → X) (F : Set X) {U : Set (X × X)}
    (U_rfl : idRel ⊆ U) (U_symm : SymmetricRel U) :
    coverEntropyEntourage T F (U ○ U) ≤ netEntropyEntourage T F U := by
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    U_rfl : HasSubset.Subset idRel U
    U_symm : SymmetricRel U
    ⊢ LE.le (Dynamics.coverEntropyEntourage T F (compRel U U)) (Dynamics.netEntrop …
  -/
  refine (limsup_le_limsup) (Eventually.of_forall fun n ↦ ?_)
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    U_rfl : HasSubset.Subset idRel U
    U_symm : SymmetricRel U
    n : Nat
    ⊢ LE.le ((fun n => HDiv.hDiv (↑(Dynamics.coverMincard T F (compRel U U) n)).lo …
  -/
  apply div_le_div_right_of_nonneg (Nat.cast_nonneg' n) (log_monotone _)
  /-
    X : Type u_1
    T : X → X
    F : Set X
    U : Set (Prod X X)
    U_rfl : HasSubset.Subset idRel U
    U_symm : SymmetricRel U
    n : Nat
    ⊢ LE.le ↑(Dynamics.coverMincard T F (compRel U U) n) ↑(Dynamics.netMaxcard T F …
  -/
  exact ENat.toENNReal_le.2 (coverMincard_le_netMaxcard T F U_rfl U_symm n)
  /-
    🎉 no goals
  -/


/-- Bowen-Dinaburg's definition of topological entropy using nets is
  `⨆ U ∈ 𝓤 X, netEntropyEntourage T F U`. This quantity is the same as the topological entropy using
  covers, so there is no need to define a new notion of topological entropy. This version of the
  theorem relates the `liminf` versions of topological entropy.-/
theorem coverEntropyInf_eq_iSup_netEntropyInfEntourage :
    coverEntropyInf T F = ⨆ U ∈ 𝓤 X, netEntropyInfEntourage T F U := by
  /-
    X : Type u_1
    inst✝ : UniformSpace X
    T : X → X
    F : Set X
    ⊢ Eq (Dynamics.coverEntropyInf T F) (iSup fun U => iSup fun h => Dynamics.netE …
  -/
  apply le_antisymm <;> refine iSup₂_le fun U U_uni ↦ ?_
    /-
      case a
      X : Type u_1
      inst✝ : UniformSpace X
      T : X → X
      F : Set X
      U : Set (Prod X X)
      U_uni : Membership.mem (uniformity X) U
      ⊢ LE.le (Dynamics.coverEntropyInfEntourage T F U) (iSup fun U => iSup fun h => …
    -/
  · rcases (comp_symm_mem_uniformity_sets U_uni) with ⟨V, V_uni, V_symm, V_comp_U⟩
    /-
      case a.intro.intro.intro
      X : Type u_1
      inst✝ : UniformSpace X
      T : X → X
      F : Set X
      U : Set (Prod X X)
      U_uni : Membership.mem (uniformity X) U
      V : Set (Prod X X)
      V_uni : Membership.mem (uniformity X) V
      V_symm : SymmetricRel V
      V_comp_U : HasSubset.Subset (compRel V V) U
      ⊢ LE.le (Dynamics.coverEntropyInfEntourage T F U) (iSup fun U => iSup fun h => …
    -/
    apply (coverEntropyInfEntourage_antitone T F V_comp_U).trans (le_iSup₂_of_le V V_uni _)
    /-
      X : Type u_1
      inst✝ : UniformSpace X
      T : X → X
      F : Set X
      U : Set (Prod X X)
      U_uni : Membership.mem (uniformity X) U
      V : Set (Prod X X)
      V_uni : Membership.mem (uniformity X) V
      V_symm : SymmetricRel V
      V_comp_U : HasSubset.Subset (compRel V V) U
      ⊢ LE.le ((fun U => Dynamics.coverEntropyInfEntourage T F U) (compRel V V)) (Dy …
    -/
    exact coverEntropyInfEntourage_le_netEntropyInfEntourage T F (refl_le_uniformity V_uni) V_symm
    /-
      🎉 no goals
    -/
    /-
      case a
      X : Type u_1
      inst✝ : UniformSpace X
      T : X → X
      F : Set X
      U : Set (Prod X X)
      U_uni : Membership.mem (uniformity X) U
      ⊢ LE.le (Dynamics.netEntropyInfEntourage T F U) (Dynamics.coverEntropyInf T F)
    -/
  · apply (netEntropyInfEntourage_antitone T F (symmetrizeRel_subset_self U)).trans
    /-
      case a
      X : Type u_1
      inst✝ : UniformSpace X
      T : X → X
      F : Set X
      U : Set (Prod X X)
      U_uni : Membership.mem (uniformity X) U
      ⊢ LE.le ((fun U => Dynamics.netEntropyInfEntourage T F U) (symmetrizeRel U)) ( …
    -/
    apply (le_iSup₂ (symmetrizeRel U) (symmetrize_mem_uniformity U_uni)).trans'
    /-
      case a
      X : Type u_1
      inst✝ : UniformSpace X
      T : X → X
      F : Set X
      U : Set (Prod X X)
      U_uni : Membership.mem (uniformity X) U
      ⊢ LE.le ((fun U => Dynamics.netEntropyInfEntourage T F U) (symmetrizeRel U)) ( …
    -/
    exact netEntropyInfEntourage_le_coverEntropyInfEntourage T F (symmetric_symmetrizeRel U)
    /-
      🎉 no goals
    -/


/-- Bowen-Dinaburg's definition of topological entropy using nets is
  `⨆ U ∈ 𝓤 X, netEntropyEntourage T F U`. This quantity is the same as the topological entropy using
  covers, so there is no need to define a new notion of topological entropy. This version of the
  theorem relates the `limsup` versions of topological entropy.-/
theorem coverEntropy_eq_iSup_netEntropyEntourage :
    coverEntropy T F = ⨆ U ∈ 𝓤 X, netEntropyEntourage T F U := by
  /-
    X : Type u_1
    inst✝ : UniformSpace X
    T : X → X
    F : Set X
    ⊢ Eq (Dynamics.coverEntropy T F) (iSup fun U => iSup fun h => Dynamics.netEntr …
  -/
  apply le_antisymm <;> refine iSup₂_le fun U U_uni ↦ ?_
    /-
      case a
      X : Type u_1
      inst✝ : UniformSpace X
      T : X → X
      F : Set X
      U : Set (Prod X X)
      U_uni : Membership.mem (uniformity X) U
      ⊢ LE.le (Dynamics.coverEntropyEntourage T F U) (iSup fun U => iSup fun h => Dy …
    -/
  · rcases (comp_symm_mem_uniformity_sets U_uni) with ⟨V, V_uni, V_symm, V_comp_U⟩
    /-
      case a.intro.intro.intro
      X : Type u_1
      inst✝ : UniformSpace X
      T : X → X
      F : Set X
      U : Set (Prod X X)
      U_uni : Membership.mem (uniformity X) U
      V : Set (Prod X X)
      V_uni : Membership.mem (uniformity X) V
      V_symm : SymmetricRel V
      V_comp_U : HasSubset.Subset (compRel V V) U
      ⊢ LE.le (Dynamics.coverEntropyEntourage T F U) (iSup fun U => iSup fun h => Dy …
    -/
    apply (coverEntropyEntourage_antitone T F V_comp_U).trans (le_iSup₂_of_le V V_uni _)
    /-
      X : Type u_1
      inst✝ : UniformSpace X
      T : X → X
      F : Set X
      U : Set (Prod X X)
      U_uni : Membership.mem (uniformity X) U
      V : Set (Prod X X)
      V_uni : Membership.mem (uniformity X) V
      V_symm : SymmetricRel V
      V_comp_U : HasSubset.Subset (compRel V V) U
      ⊢ LE.le ((fun U => Dynamics.coverEntropyEntourage T F U) (compRel V V)) (Dynam …
    -/
    exact coverEntropyEntourage_le_netEntropyEntourage T F (refl_le_uniformity V_uni) V_symm
    /-
      🎉 no goals
    -/
    /-
      case a
      X : Type u_1
      inst✝ : UniformSpace X
      T : X → X
      F : Set X
      U : Set (Prod X X)
      U_uni : Membership.mem (uniformity X) U
      ⊢ LE.le (Dynamics.netEntropyEntourage T F U) (Dynamics.coverEntropy T F)
    -/
  · apply (netEntropyEntourage_antitone T F (symmetrizeRel_subset_self U)).trans
    /-
      case a
      X : Type u_1
      inst✝ : UniformSpace X
      T : X → X
      F : Set X
      U : Set (Prod X X)
      U_uni : Membership.mem (uniformity X) U
      ⊢ LE.le ((fun U => Dynamics.netEntropyEntourage T F U) (symmetrizeRel U)) (Dyn …
    -/
    apply (le_iSup₂ (symmetrizeRel U) (symmetrize_mem_uniformity U_uni)).trans'
    /-
      case a
      X : Type u_1
      inst✝ : UniformSpace X
      T : X → X
      F : Set X
      U : Set (Prod X X)
      U_uni : Membership.mem (uniformity X) U
      ⊢ LE.le ((fun U => Dynamics.netEntropyEntourage T F U) (symmetrizeRel U)) (Dyn …
    -/
    exact netEntropyEntourage_le_coverEntropyEntourage T F (symmetric_symmetrizeRel U)
    /-
      🎉 no goals
    -/


lemma coverEntropyInf_eq_iSup_basis_netEntropyInfEntourage {ι : Sort*} {p : ι → Prop}
    {s : ι → Set (X × X)} (h : (𝓤 X).HasBasis p s) (T : X → X) (F : Set X) :
    coverEntropyInf T F = ⨆ (i : ι) (_ : p i), netEntropyInfEntourage T F (s i) := by
  /-
    X : Type u_1
    inst✝ : UniformSpace X
    ι : Sort u_2
    p : ι → Prop
    s : ι → Set (Prod X X)
    h : (uniformity X).HasBasis p s
    T : X → X
    F : Set X
    ⊢ Eq (Dynamics.coverEntropyInf T F) (iSup fun i => iSup fun x => Dynamics.netE …
  -/
  rw [coverEntropyInf_eq_iSup_netEntropyInfEntourage T F]
  /-
    X : Type u_1
    inst✝ : UniformSpace X
    ι : Sort u_2
    p : ι → Prop
    s : ι → Set (Prod X X)
    h : (uniformity X).HasBasis p s
    T : X → X
    F : Set X
    ⊢ Eq (iSup fun U => iSup fun h => Dynamics.netEntropyInfEntourage T F U) (iSup …
  -/
  apply (iSup₂_mono' fun i h_i ↦ ⟨s i, HasBasis.mem_of_mem h h_i, le_refl _⟩).antisymm'
  /-
    X : Type u_1
    inst✝ : UniformSpace X
    ι : Sort u_2
    p : ι → Prop
    s : ι → Set (Prod X X)
    h : (uniformity X).HasBasis p s
    T : X → X
    F : Set X
    ⊢ LE.le (iSup fun i => iSup fun j => Dynamics.netEntropyInfEntourage T F i) (i …
  -/
  refine iSup₂_le fun U U_uni ↦ ?_
  /-
    X : Type u_1
    inst✝ : UniformSpace X
    ι : Sort u_2
    p : ι → Prop
    s : ι → Set (Prod X X)
    h : (uniformity X).HasBasis p s
    T : X → X
    F : Set X
    U : Set (Prod X X)
    U_uni : Membership.mem (uniformity X) U
    ⊢ LE.le (Dynamics.netEntropyInfEntourage T F U) (iSup fun i => iSup fun j => D …
  -/
  rcases (HasBasis.mem_iff h).1 U_uni with ⟨i, h_i, si_U⟩
  /-
    case intro.intro
    X : Type u_1
    inst✝ : UniformSpace X
    ι : Sort u_2
    p : ι → Prop
    s : ι → Set (Prod X X)
    h : (uniformity X).HasBasis p s
    T : X → X
    F : Set X
    U : Set (Prod X X)
    U_uni : Membership.mem (uniformity X) U
    i : ι
    h_i : p i
    si_U : HasSubset.Subset (s i) U
    ⊢ LE.le (Dynamics.netEntropyInfEntourage T F U) (iSup fun i => iSup fun j => D …
  -/
  apply (netEntropyInfEntourage_antitone T F si_U).trans
  /-
    case intro.intro
    X : Type u_1
    inst✝ : UniformSpace X
    ι : Sort u_2
    p : ι → Prop
    s : ι → Set (Prod X X)
    h : (uniformity X).HasBasis p s
    T : X → X
    F : Set X
    U : Set (Prod X X)
    U_uni : Membership.mem (uniformity X) U
    i : ι
    h_i : p i
    si_U : HasSubset.Subset (s i) U
    ⊢ LE.le ((fun U => Dynamics.netEntropyInfEntourage T F U) (s i)) (iSup fun i = …
  -/
  exact le_iSup₂ (f := fun (i : ι) (_ : p i) ↦ netEntropyInfEntourage T F (s i)) i h_i
  /-
    🎉 no goals
  -/


lemma coverEntropy_eq_iSup_basis_netEntropyEntourage {ι : Sort*} {p : ι → Prop}
    {s : ι → Set (X × X)} (h : (𝓤 X).HasBasis p s) (T : X → X) (F : Set X) :
    coverEntropy T F = ⨆ (i : ι) (_ : p i), netEntropyEntourage T F (s i) := by
  /-
    X : Type u_1
    inst✝ : UniformSpace X
    ι : Sort u_2
    p : ι → Prop
    s : ι → Set (Prod X X)
    h : (uniformity X).HasBasis p s
    T : X → X
    F : Set X
    ⊢ Eq (Dynamics.coverEntropy T F) (iSup fun i => iSup fun x => Dynamics.netEntr …
  -/
  rw [coverEntropy_eq_iSup_netEntropyEntourage T F]
  /-
    X : Type u_1
    inst✝ : UniformSpace X
    ι : Sort u_2
    p : ι → Prop
    s : ι → Set (Prod X X)
    h : (uniformity X).HasBasis p s
    T : X → X
    F : Set X
    ⊢ Eq (iSup fun U => iSup fun h => Dynamics.netEntropyEntourage T F U) (iSup fu …
  -/
  apply (iSup₂_mono' fun i h_i ↦ ⟨s i, HasBasis.mem_of_mem h h_i, le_refl _⟩).antisymm'
  /-
    X : Type u_1
    inst✝ : UniformSpace X
    ι : Sort u_2
    p : ι → Prop
    s : ι → Set (Prod X X)
    h : (uniformity X).HasBasis p s
    T : X → X
    F : Set X
    ⊢ LE.le (iSup fun i => iSup fun j => Dynamics.netEntropyEntourage T F i) (iSup …
  -/
  refine iSup₂_le fun U U_uni ↦ ?_
  /-
    X : Type u_1
    inst✝ : UniformSpace X
    ι : Sort u_2
    p : ι → Prop
    s : ι → Set (Prod X X)
    h : (uniformity X).HasBasis p s
    T : X → X
    F : Set X
    U : Set (Prod X X)
    U_uni : Membership.mem (uniformity X) U
    ⊢ LE.le (Dynamics.netEntropyEntourage T F U) (iSup fun i => iSup fun j => Dyna …
  -/
  rcases (HasBasis.mem_iff h).1 U_uni with ⟨i, h_i, si_U⟩
  /-
    case intro.intro
    X : Type u_1
    inst✝ : UniformSpace X
    ι : Sort u_2
    p : ι → Prop
    s : ι → Set (Prod X X)
    h : (uniformity X).HasBasis p s
    T : X → X
    F : Set X
    U : Set (Prod X X)
    U_uni : Membership.mem (uniformity X) U
    i : ι
    h_i : p i
    si_U : HasSubset.Subset (s i) U
    ⊢ LE.le (Dynamics.netEntropyEntourage T F U) (iSup fun i => iSup fun j => Dyna …
  -/
  apply (netEntropyEntourage_antitone T F si_U).trans _
  /-
    X : Type u_1
    inst✝ : UniformSpace X
    ι : Sort u_2
    p : ι → Prop
    s : ι → Set (Prod X X)
    h : (uniformity X).HasBasis p s
    T : X → X
    F : Set X
    U : Set (Prod X X)
    U_uni : Membership.mem (uniformity X) U
    i : ι
    h_i : p i
    si_U : HasSubset.Subset (s i) U
    ⊢ LE.le ((fun U => Dynamics.netEntropyEntourage T F U) (s i)) (iSup fun i => i …
  -/
  exact le_iSup₂ (f := fun (i : ι) (_ : p i) ↦ netEntropyEntourage T F (s i)) i h_i
  /-
    🎉 no goals
  -/


lemma netEntropyInfEntourage_le_coverEntropyInf {U : Set (X × X)} (h : U ∈ 𝓤 X) :
    netEntropyInfEntourage T F U ≤ coverEntropyInf T F :=
  coverEntropyInf_eq_iSup_netEntropyInfEntourage T F ▸
    le_iSup₂ (f := fun (U : Set (X × X)) (_ : U ∈ 𝓤 X) ↦ netEntropyInfEntourage T F U) U h


lemma netEntropyEntourage_le_coverEntropy {U : Set (X × X)} (h : U ∈ 𝓤 X) :
    netEntropyEntourage T F U ≤ coverEntropy T F :=
  coverEntropy_eq_iSup_netEntropyEntourage T F ▸
    le_iSup₂ (f := fun (U : Set (X × X)) (_ : U ∈ 𝓤 X) ↦ netEntropyEntourage T F U) U h



/-- Basis for the topology on `Ultrafilter α`. -/
def ultrafilterBasis (α : Type u) : Set (Set (Ultrafilter α)) :=
  range fun s : Set α ↦ { u | s ∈ u }


instance Ultrafilter.topologicalSpace : TopologicalSpace (Ultrafilter α) :=
  TopologicalSpace.generateFrom (ultrafilterBasis α)


theorem ultrafilterBasis_is_basis : TopologicalSpace.IsTopologicalBasis (ultrafilterBasis α) :=
  ⟨by
    /-
      α : Type u
      ⊢ ∀ (t₁ : Set (Ultrafilter α)), Membership.mem (ultrafilterBasis α) t₁ → ∀ (t₂ …
    -/
    rintro _ ⟨a, rfl⟩ _ ⟨b, rfl⟩ u ⟨ua, ub⟩
    /-
      case intro.intro.intro
      α : Type u
      a b : Set α
      u : Ultrafilter α
      ua : Membership.mem ((fun s => setOf fun u => Membership.mem u s) a) u
      ub : Membership.mem ((fun s => setOf fun u => Membership.mem u s) b) u
      ⊢ Exists fun t₃ => And (Membership.mem (ultrafilterBasis α) t₃) (And (Membersh …
    -/
    refine ⟨_, ⟨a ∩ b, rfl⟩, inter_mem ua ub, fun v hv ↦ ⟨?_, ?_⟩⟩ <;> apply mem_of_superset hv <;>
      /-
        case intro.intro.intro.refine_1
        α : Type u
        a b : Set α
        u : Ultrafilter α
        ua : Membership.mem ((fun s => setOf fun u => Membership.mem u s) a) u
        ub : Membership.mem ((fun s => setOf fun u => Membership.mem u s) b) u
        v : Ultrafilter α
        hv : Membership.mem ((fun s => setOf fun u => Membership.mem u s) (Inter.inter …
        ⊢ HasSubset.Subset (Inter.inter a b) a
      -/
      /-
        🎉 no goals
      -/
      simp [inter_subset_right],
      /-
        🎉 no goals
      -/
    eq_univ_of_univ_subset <| subset_sUnion_of_mem <| ⟨univ, eq_univ_of_forall fun _ ↦ univ_mem⟩,
    rfl⟩


/-- The basic open sets for the topology on ultrafilters are open. -/
theorem ultrafilter_isOpen_basic (s : Set α) : IsOpen { u : Ultrafilter α | s ∈ u } :=
  ultrafilterBasis_is_basis.isOpen ⟨s, rfl⟩


/-- The basic open sets for the topology on ultrafilters are also closed. -/
theorem ultrafilter_isClosed_basic (s : Set α) : IsClosed { u : Ultrafilter α | s ∈ u } := by
  /-
    α : Type u
    s : Set α
    ⊢ IsClosed (setOf fun u => Membership.mem u s)
  -/
  rw [← isOpen_compl_iff]
  /-
    α : Type u
    s : Set α
    ⊢ IsOpen (HasCompl.compl (setOf fun u => Membership.mem u s))
  -/
  convert ultrafilter_isOpen_basic sᶜ using 1
  /-
    case h.e'_3
    α : Type u
    s : Set α
    ⊢ Eq (HasCompl.compl (setOf fun u => Membership.mem u s)) (setOf fun u => Memb …
  -/
  ext u
  /-
    case h.e'_3.h
    α : Type u
    s : Set α
    u : Ultrafilter α
    ⊢ Iff (Membership.mem (HasCompl.compl (setOf fun u => Membership.mem u s)) u)  …
  -/
  exact Ultrafilter.compl_mem_iff_not_mem.symm
  /-
    🎉 no goals
  -/


/-- Every ultrafilter `u` on `Ultrafilter α` converges to a unique
  point of `Ultrafilter α`, namely `joinM u`. -/
theorem ultrafilter_converges_iff {u : Ultrafilter (Ultrafilter α)} {x : Ultrafilter α} :
    ↑u ≤ 𝓝 x ↔ x = joinM u := by
  /-
    α : Type u
    u : Ultrafilter (Ultrafilter α)
    x : Ultrafilter α
    ⊢ Iff (LE.le (↑u) (nhds x)) (Eq x (joinM u))
  -/
  rw [eq_comm, ← Ultrafilter.coe_le_coe]
  /-
    α : Type u
    u : Ultrafilter (Ultrafilter α)
    x : Ultrafilter α
    ⊢ Iff (LE.le (↑u) (nhds x)) (LE.le ↑(joinM u) ↑x)
  -/
  change ↑u ≤ 𝓝 x ↔ ∀ s ∈ x, { v : Ultrafilter α | s ∈ v } ∈ u
  simp only [TopologicalSpace.nhds_generateFrom, le_iInf_iff, ultrafilterBasis, le_principal_iff,
    mem_setOf_eq]
  /-
    α : Type u
    u : Ultrafilter (Ultrafilter α)
    x : Ultrafilter α
    ⊢ Iff (∀ (i : Set (Ultrafilter α)), And (Membership.mem i x) (Membership.mem ( …
  -/
  constructor
    /-
      case mp
      α : Type u
      u : Ultrafilter (Ultrafilter α)
      x : Ultrafilter α
      ⊢ (∀ (i : Set (Ultrafilter α)), And (Membership.mem i x) (Membership.mem (Set. …
    -/
  · intro h a ha
    /-
      case mp
      α : Type u
      u : Ultrafilter (Ultrafilter α)
      x : Ultrafilter α
      h : ∀ (i : Set (Ultrafilter α)), And (Membership.mem i x) (Membership.mem (Set …
      a : Set α
      ha : Membership.mem x a
      ⊢ Membership.mem u (setOf fun v => Membership.mem v a)
    -/
    exact h _ ⟨ha, a, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      u : Ultrafilter (Ultrafilter α)
      x : Ultrafilter α
      ⊢ (∀ (s : Set α), Membership.mem x s → Membership.mem u (setOf fun v => Member …
    -/
  · rintro h a ⟨xi, a, rfl⟩
    /-
      case mpr.intro.intro
      α : Type u
      u : Ultrafilter (Ultrafilter α)
      x : Ultrafilter α
      h : ∀ (s : Set α), Membership.mem x s → Membership.mem u (setOf fun v => Membe …
      a : Set α
      xi : Membership.mem ((fun s => setOf fun u => Membership.mem u s) a) x
      ⊢ Membership.mem (↑u) ((fun s => setOf fun u => Membership.mem u s) a)
    -/
    exact h _ xi
    /-
      🎉 no goals
    -/


instance ultrafilter_compact : CompactSpace (Ultrafilter α) :=
  ⟨isCompact_iff_ultrafilter_le_nhds.mpr fun f _ ↦
      ⟨joinM f, trivial, ultrafilter_converges_iff.mpr rfl⟩⟩


instance Ultrafilter.t2Space : T2Space (Ultrafilter α) :=
  t2_iff_ultrafilter.mpr fun {x y} f fx fy ↦
    have hx : x = joinM f := ultrafilter_converges_iff.mp fx
    have hy : y = joinM f := ultrafilter_converges_iff.mp fy
    hx.trans hy.symm


instance : TotallyDisconnectedSpace (Ultrafilter α) := by
  /-
    α : Type u
    ⊢ TotallyDisconnectedSpace (Ultrafilter α)
  -/
  rw [totallyDisconnectedSpace_iff_connectedComponent_singleton]
  /-
    α : Type u
    ⊢ ∀ (x : Ultrafilter α), Eq (connectedComponent x) (Singleton.singleton x)
  -/
  intro A
  /-
    α : Type u
    A : Ultrafilter α
    ⊢ Eq (connectedComponent A) (Singleton.singleton A)
  -/
  simp only [Set.eq_singleton_iff_unique_mem, mem_connectedComponent, true_and]
  /-
    α : Type u
    A : Ultrafilter α
    ⊢ ∀ (x : Ultrafilter α), Membership.mem (connectedComponent A) x → Eq x A
  -/
  intro B hB
  /-
    α : Type u
    A B : Ultrafilter α
    hB : Membership.mem (connectedComponent A) B
    ⊢ Eq B A
  -/
  rw [← Ultrafilter.coe_le_coe]
  /-
    α : Type u
    A B : Ultrafilter α
    hB : Membership.mem (connectedComponent A) B
    ⊢ LE.le ↑B ↑A
  -/
  intro s hs
  /-
    α : Type u
    A B : Ultrafilter α
    hB : Membership.mem (connectedComponent A) B
    s : Set α
    hs : Membership.mem (↑A) s
    ⊢ Membership.mem (↑B) s
  -/
  rw [connectedComponent_eq_iInter_isClopen, Set.mem_iInter] at hB
  /-
    α : Type u
    A B : Ultrafilter α
    hB : ∀ (i : Subtype fun s => And (IsClopen s) (Membership.mem s A)), Membershi …
    s : Set α
    hs : Membership.mem (↑A) s
    ⊢ Membership.mem (↑B) s
  -/
  let Z := { F : Ultrafilter α | s ∈ F }
  /-
    α : Type u
    A B : Ultrafilter α
    hB : ∀ (i : Subtype fun s => And (IsClopen s) (Membership.mem s A)), Membershi …
    s : Set α
    hs : Membership.mem (↑A) s
    Z : Set (Ultrafilter α) := setOf fun F => Membership.mem F s
    ⊢ Membership.mem (↑B) s
  -/
  have hZ : IsClopen Z := ⟨ultrafilter_isClosed_basic s, ultrafilter_isOpen_basic s⟩
  /-
    α : Type u
    A B : Ultrafilter α
    hB : ∀ (i : Subtype fun s => And (IsClopen s) (Membership.mem s A)), Membershi …
    s : Set α
    hs : Membership.mem (↑A) s
    Z : Set (Ultrafilter α) := setOf fun F => Membership.mem F s
    hZ : IsClopen Z
    ⊢ Membership.mem (↑B) s
  -/
  exact hB ⟨Z, hZ, hs⟩
  /-
    🎉 no goals
  -/


@[simp] theorem Ultrafilter.tendsto_pure_self (b : Ultrafilter α) : Tendsto pure b (𝓝 b) := by
  /-
    α : Type u
    b : Ultrafilter α
    ⊢ Filter.Tendsto Pure.pure (↑b) (nhds b)
  -/
  rw [Tendsto, ← coe_map, ultrafilter_converges_iff]
  /-
    α : Type u
    b : Ultrafilter α
    ⊢ Eq b (joinM (Ultrafilter.map Pure.pure b))
  -/
  ext s
  /-
    case h
    α : Type u
    b : Ultrafilter α
    s : Set α
    ⊢ Iff (Membership.mem b s) (Membership.mem (joinM (Ultrafilter.map Pure.pure b …
  -/
  change s ∈ b ↔ {t | s ∈ t} ∈ map pure b
  /-
    case h
    α : Type u
    b : Ultrafilter α
    s : Set α
    ⊢ Iff (Membership.mem b s) (Membership.mem (Ultrafilter.map Pure.pure b) (setO …
  -/
  simp_rw [mem_map, preimage_setOf_eq, mem_pure, setOf_mem_eq]
  /-
    🎉 no goals
  -/


theorem ultrafilter_comap_pure_nhds (b : Ultrafilter α) : comap pure (𝓝 b) ≤ b := by
  /-
    α : Type u
    b : Ultrafilter α
    ⊢ LE.le (Filter.comap Pure.pure (nhds b)) ↑b
  -/
  rw [TopologicalSpace.nhds_generateFrom]
  /-
    α : Type u
    b : Ultrafilter α
    ⊢ LE.le (Filter.comap Pure.pure (iInf fun s => iInf fun h => Filter.principal  …
  -/
  simp only [comap_iInf, comap_principal]
  /-
    α : Type u
    b : Ultrafilter α
    ⊢ LE.le (iInf fun i => iInf fun x => Filter.principal (Set.preimage Pure.pure  …
  -/
  intro s hs
  /-
    α : Type u
    b : Ultrafilter α
    s : Set α
    hs : Membership.mem (↑b) s
    ⊢ Membership.mem (iInf fun i => iInf fun x => Filter.principal (Set.preimage P …
  -/
  rw [← le_principal_iff]
  /-
    α : Type u
    b : Ultrafilter α
    s : Set α
    hs : Membership.mem (↑b) s
    ⊢ LE.le (iInf fun i => iInf fun x => Filter.principal (Set.preimage Pure.pure  …
  -/
  refine iInf_le_of_le { u | s ∈ u } ?_
  /-
    α : Type u
    b : Ultrafilter α
    s : Set α
    hs : Membership.mem (↑b) s
    ⊢ LE.le (iInf fun x => Filter.principal (Set.preimage Pure.pure (setOf fun u = …
  -/
  refine iInf_le_of_le ⟨hs, ⟨s, rfl⟩⟩ ?_
  /-
    α : Type u
    b : Ultrafilter α
    s : Set α
    hs : Membership.mem (↑b) s
    ⊢ LE.le (Filter.principal (Set.preimage Pure.pure (setOf fun u => Membership.m …
  -/
  exact principal_mono.2 fun _ ↦ id
  /-
    🎉 no goals
  -/


theorem ultrafilter_pure_injective : Function.Injective (pure : α → Ultrafilter α) := by
  /-
    α : Type u
    ⊢ Function.Injective Pure.pure
  -/
  intro x y h
  /-
    α : Type u
    x y : α
    h : Eq (Pure.pure x) (Pure.pure y)
    ⊢ Eq x y
  -/
  have : {x} ∈ (pure x : Ultrafilter α) := singleton_mem_pure
  /-
    α : Type u
    x y : α
    h : Eq (Pure.pure x) (Pure.pure y)
    this : Membership.mem (Pure.pure x) (Singleton.singleton x)
    ⊢ Eq x y
  -/
  rw [h] at this
  /-
    α : Type u
    x y : α
    h : Eq (Pure.pure x) (Pure.pure y)
    this : Membership.mem (Pure.pure y) (Singleton.singleton x)
    ⊢ Eq x y
  -/
  exact (mem_singleton_iff.mp (mem_pure.mp this)).symm
  /-
    🎉 no goals
  -/


/-- The range of `pure : α → Ultrafilter α` is dense in `Ultrafilter α`. -/
theorem denseRange_pure : DenseRange (pure : α → Ultrafilter α) :=
  fun x ↦ mem_closure_iff_ultrafilter.mpr
    ⟨x.map pure, range_mem_map, ultrafilter_converges_iff.mpr (bind_pure x).symm⟩


/-- The map `pure : α → Ultrafilter α` induces on `α` the discrete topology. -/
theorem induced_topology_pure :
    TopologicalSpace.induced (pure : α → Ultrafilter α) Ultrafilter.topologicalSpace = ⊥ := by
  /-
    α : Type u
    ⊢ Eq (TopologicalSpace.induced Pure.pure Ultrafilter.topologicalSpace) Bot.bot
  -/
  apply eq_bot_of_singletons_open
  /-
    case h
    α : Type u
    ⊢ ∀ (x : α), IsOpen (Singleton.singleton x)
  -/
  intro x
  /-
    case h
    α : Type u
    x : α
    ⊢ IsOpen (Singleton.singleton x)
  -/
  use { u : Ultrafilter α | {x} ∈ u }, ultrafilter_isOpen_basic _
  /-
    case right
    α : Type u
    x : α
    ⊢ Eq (Set.preimage Pure.pure (setOf fun u => Membership.mem u (Singleton.singl …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- `pure : α → Ultrafilter α` defines a dense inducing of `α` in `Ultrafilter α`. -/
theorem isDenseInducing_pure : @IsDenseInducing _ _ ⊥ _ (pure : α → Ultrafilter α) :=
  letI : TopologicalSpace α := ⊥
  ⟨⟨induced_topology_pure.symm⟩, denseRange_pure⟩

-- The following refined version will never be used

/-- `pure : α → Ultrafilter α` defines a dense embedding of `α` in `Ultrafilter α`. -/
theorem isDenseEmbedding_pure : @IsDenseEmbedding _ _ ⊥ _ (pure : α → Ultrafilter α) :=
  letI : TopologicalSpace α := ⊥
  { isDenseInducing_pure with injective := ultrafilter_pure_injective }


@[deprecated (since := "2024-09-30")]
alias denseEmbedding_pure := isDenseEmbedding_pure


/-- The extension of a function `α → γ` to a function `Ultrafilter α → γ`.
  When `γ` is a compact Hausdorff space it will be continuous. -/
def Ultrafilter.extend (f : α → γ) : Ultrafilter α → γ :=
  letI : TopologicalSpace α := ⊥
  isDenseInducing_pure.extend f


theorem ultrafilter_extend_extends (f : α → γ) : Ultrafilter.extend f ∘ pure = f := by
  /-
    α : Type u
    γ : Type u_1
    inst✝¹ : TopologicalSpace γ
    inst✝ : T2Space γ
    f : α → γ
    ⊢ Eq (Function.comp (Ultrafilter.extend f) Pure.pure) f
  -/
  letI : TopologicalSpace α := ⊥
  /-
    α : Type u
    γ : Type u_1
    inst✝¹ : TopologicalSpace γ
    inst✝ : T2Space γ
    f : α → γ
    this : TopologicalSpace α := Bot.bot
    ⊢ Eq (Function.comp (Ultrafilter.extend f) Pure.pure) f
  -/
  haveI : DiscreteTopology α := ⟨rfl⟩
  /-
    α : Type u
    γ : Type u_1
    inst✝¹ : TopologicalSpace γ
    inst✝ : T2Space γ
    f : α → γ
    this✝ : TopologicalSpace α := Bot.bot
    this : DiscreteTopology α
    ⊢ Eq (Function.comp (Ultrafilter.extend f) Pure.pure) f
  -/
  exact funext (isDenseInducing_pure.extend_eq continuous_of_discreteTopology)
  /-
    🎉 no goals
  -/


theorem continuous_ultrafilter_extend (f : α → γ) : Continuous (Ultrafilter.extend f) := by
  have h (b : Ultrafilter α) : ∃ c, Tendsto f (comap pure (𝓝 b)) (𝓝 c) :=
    -- b.map f is an ultrafilter on γ, which is compact, so it converges to some c in γ.
    let ⟨c, _, h'⟩ :=
      isCompact_univ.ultrafilter_le_nhds (b.map f) (by rw [le_principal_iff]; exact univ_mem)
    ⟨c, le_trans (map_mono (ultrafilter_comap_pure_nhds _)) h'⟩
  /-
    α : Type u
    γ : Type u_1
    inst✝² : TopologicalSpace γ
    inst✝¹ : T2Space γ
    inst✝ : CompactSpace γ
    f : α → γ
    h : ∀ (b : Ultrafilter α), Exists fun c => Filter.Tendsto f (Filter.comap Pure …
    ⊢ Continuous (Ultrafilter.extend f)
  -/
  let _ : TopologicalSpace α := ⊥
  /-
    α : Type u
    γ : Type u_1
    inst✝² : TopologicalSpace γ
    inst✝¹ : T2Space γ
    inst✝ : CompactSpace γ
    f : α → γ
    h : ∀ (b : Ultrafilter α), Exists fun c => Filter.Tendsto f (Filter.comap Pure …
    x✝ : TopologicalSpace α := Bot.bot
    ⊢ Continuous (Ultrafilter.extend f)
  -/
  exact isDenseInducing_pure.continuous_extend h
  /-
    🎉 no goals
  -/


/-- The value of `Ultrafilter.extend f` on an ultrafilter `b` is the
  unique limit of the ultrafilter `b.map f` in `γ`. -/
theorem ultrafilter_extend_eq_iff {f : α → γ} {b : Ultrafilter α} {c : γ} :
    Ultrafilter.extend f b = c ↔ ↑(b.map f) ≤ 𝓝 c :=
  ⟨fun h ↦ by
     -- Write b as an ultrafilter limit of pure ultrafilters, and use
     -- the facts that ultrafilter.extend is a continuous extension of f.
     /-
       α : Type u
       γ : Type u_1
       inst✝² : TopologicalSpace γ
       inst✝¹ : T2Space γ
       inst✝ : CompactSpace γ
       f : α → γ
       b : Ultrafilter α
       c : γ
       h : Eq (Ultrafilter.extend f b) c
       ⊢ LE.le (↑(Ultrafilter.map f b)) (nhds c)
     -/
     let b' : Ultrafilter (Ultrafilter α) := b.map pure
     /-
       α : Type u
       γ : Type u_1
       inst✝² : TopologicalSpace γ
       inst✝¹ : T2Space γ
       inst✝ : CompactSpace γ
       f : α → γ
       b : Ultrafilter α
       c : γ
       h : Eq (Ultrafilter.extend f b) c
       b' : Ultrafilter (Ultrafilter α) := Ultrafilter.map Pure.pure b
       ⊢ LE.le (↑(Ultrafilter.map f b)) (nhds c)
     -/
     have t : ↑b' ≤ 𝓝 b := ultrafilter_converges_iff.mpr (bind_pure _).symm
     /-
       α : Type u
       γ : Type u_1
       inst✝² : TopologicalSpace γ
       inst✝¹ : T2Space γ
       inst✝ : CompactSpace γ
       f : α → γ
       b : Ultrafilter α
       c : γ
       h : Eq (Ultrafilter.extend f b) c
       b' : Ultrafilter (Ultrafilter α) := Ultrafilter.map Pure.pure b
       t : LE.le (↑b') (nhds b)
       ⊢ LE.le (↑(Ultrafilter.map f b)) (nhds c)
     -/
     rw [← h]
     /-
       α : Type u
       γ : Type u_1
       inst✝² : TopologicalSpace γ
       inst✝¹ : T2Space γ
       inst✝ : CompactSpace γ
       f : α → γ
       b : Ultrafilter α
       c : γ
       h : Eq (Ultrafilter.extend f b) c
       b' : Ultrafilter (Ultrafilter α) := Ultrafilter.map Pure.pure b
       t : LE.le (↑b') (nhds b)
       ⊢ LE.le (↑(Ultrafilter.map f b)) (nhds (Ultrafilter.extend f b))
     -/
     have := (continuous_ultrafilter_extend f).tendsto b
     /-
       α : Type u
       γ : Type u_1
       inst✝² : TopologicalSpace γ
       inst✝¹ : T2Space γ
       inst✝ : CompactSpace γ
       f : α → γ
       b : Ultrafilter α
       c : γ
       h : Eq (Ultrafilter.extend f b) c
       b' : Ultrafilter (Ultrafilter α) := Ultrafilter.map Pure.pure b
       t : LE.le (↑b') (nhds b)
       this : Filter.Tendsto (Ultrafilter.extend f) (nhds b) (nhds (Ultrafilter.exten …
       ⊢ LE.le (↑(Ultrafilter.map f b)) (nhds (Ultrafilter.extend f b))
     -/
     refine le_trans ?_ (le_trans (map_mono t) this)
     /-
       α : Type u
       γ : Type u_1
       inst✝² : TopologicalSpace γ
       inst✝¹ : T2Space γ
       inst✝ : CompactSpace γ
       f : α → γ
       b : Ultrafilter α
       c : γ
       h : Eq (Ultrafilter.extend f b) c
       b' : Ultrafilter (Ultrafilter α) := Ultrafilter.map Pure.pure b
       t : LE.le (↑b') (nhds b)
       this : Filter.Tendsto (Ultrafilter.extend f) (nhds b) (nhds (Ultrafilter.exten …
       ⊢ LE.le (↑(Ultrafilter.map f b)) (Filter.map (Ultrafilter.extend f) ↑b')
     -/
     change _ ≤ map (Ultrafilter.extend f ∘ pure) ↑b
     /-
       α : Type u
       γ : Type u_1
       inst✝² : TopologicalSpace γ
       inst✝¹ : T2Space γ
       inst✝ : CompactSpace γ
       f : α → γ
       b : Ultrafilter α
       c : γ
       h : Eq (Ultrafilter.extend f b) c
       b' : Ultrafilter (Ultrafilter α) := Ultrafilter.map Pure.pure b
       t : LE.le (↑b') (nhds b)
       this : Filter.Tendsto (Ultrafilter.extend f) (nhds b) (nhds (Ultrafilter.exten …
       ⊢ LE.le (↑(Ultrafilter.map f b)) (Filter.map (Function.comp (Ultrafilter.exten …
     -/
     rw [ultrafilter_extend_extends]
     /-
       α : Type u
       γ : Type u_1
       inst✝² : TopologicalSpace γ
       inst✝¹ : T2Space γ
       inst✝ : CompactSpace γ
       f : α → γ
       b : Ultrafilter α
       c : γ
       h : Eq (Ultrafilter.extend f b) c
       b' : Ultrafilter (Ultrafilter α) := Ultrafilter.map Pure.pure b
       t : LE.le (↑b') (nhds b)
       this : Filter.Tendsto (Ultrafilter.extend f) (nhds b) (nhds (Ultrafilter.exten …
       ⊢ LE.le (↑(Ultrafilter.map f b)) (Filter.map f ↑b)
     -/
     exact le_rfl,
     /-
       🎉 no goals
     -/
   fun h ↦
    let _ : TopologicalSpace α := ⊥
    isDenseInducing_pure.extend_eq_of_tendsto
      (le_trans (map_mono (ultrafilter_comap_pure_nhds _)) h)⟩


/-- Auxiliary construction towards the Stone-Čech compactification of a topological space.
It should not be used after the Stone-Čech compactification is constructed. -/
def PreStoneCech : Type u :=
  Quot fun F G : Ultrafilter α ↦ ∃ x, (F : Filter α) ≤ 𝓝 x ∧ (G : Filter α) ≤ 𝓝 x


instance : TopologicalSpace (PreStoneCech α) :=
  inferInstanceAs (TopologicalSpace <| Quot _)


instance : CompactSpace (PreStoneCech α) :=
  Quot.compactSpace


instance [Inhabited α] : Inhabited (PreStoneCech α) :=
  inferInstanceAs (Inhabited <| Quot _)


/-- The natural map from α to its pre-Stone-Čech compactification. -/
def preStoneCechUnit (x : α) : PreStoneCech α :=
  Quot.mk _ (pure x : Ultrafilter α)


theorem continuous_preStoneCechUnit : Continuous (preStoneCechUnit : α → PreStoneCech α) :=
  continuous_iff_ultrafilter.mpr fun x g gx ↦ by
    have : (g.map pure).toFilter ≤ 𝓝 g := by
      rw [ultrafilter_converges_iff, ← bind_pure g]
      rfl
    have : (map preStoneCechUnit g : Filter (PreStoneCech α)) ≤ 𝓝 (Quot.mk _ g) :=
      (map_mono this).trans (continuous_quot_mk.tendsto _)
    /-
      α : Type u
      inst✝ : TopologicalSpace α
      x : α
      g : Ultrafilter α
      gx : LE.le (↑g) (nhds x)
      this✝ : LE.le (↑(Ultrafilter.map Pure.pure g)) (nhds g)
      this : LE.le (Filter.map preStoneCechUnit ↑g) (nhds (Quot.mk (fun F G => Exist …
      ⊢ Filter.Tendsto preStoneCechUnit (↑g) (nhds (preStoneCechUnit x))
    -/
    convert this
    /-
      case h.e'_1.h.e'_3.h
      α : Type u
      inst✝ : TopologicalSpace α
      x : α
      g : Ultrafilter α
      gx : LE.le (↑g) (nhds x)
      this✝ : LE.le (↑(Ultrafilter.map Pure.pure g)) (nhds g)
      this : LE.le (Filter.map preStoneCechUnit ↑g) (nhds (Quot.mk (fun F G => Exist …
      e_1✝ : Eq (PreStoneCech α) (Quot fun F G => Exists fun x => And (LE.le (↑F) (n …
      ⊢ Eq (preStoneCechUnit x) (Quot.mk (fun F G => Exists fun x => And (LE.le (↑F) …
    -/
    exact Quot.sound ⟨x, pure_le_nhds x, gx⟩
    /-
      🎉 no goals
    -/


theorem denseRange_preStoneCechUnit : DenseRange (preStoneCechUnit : α → PreStoneCech α) :=
  Quot.mk_surjective.denseRange.comp denseRange_pure continuous_coinduced_rng



theorem preStoneCech_hom_ext {g₁ g₂ : PreStoneCech α → β} (h₁ : Continuous g₁) (h₂ : Continuous g₂)
    (h : g₁ ∘ preStoneCechUnit = g₂ ∘ preStoneCechUnit) : g₁ = g₂ := by
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    β : Type v
    inst✝¹ : TopologicalSpace β
    inst✝ : T2Space β
    g₁ g₂ : PreStoneCech α → β
    h₁ : Continuous g₁
    h₂ : Continuous g₂
    h : Eq (Function.comp g₁ preStoneCechUnit) (Function.comp g₂ preStoneCechUnit)
    ⊢ Eq g₁ g₂
  -/
  apply Continuous.ext_on denseRange_preStoneCechUnit h₁ h₂
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    β : Type v
    inst✝¹ : TopologicalSpace β
    inst✝ : T2Space β
    g₁ g₂ : PreStoneCech α → β
    h₁ : Continuous g₁
    h₂ : Continuous g₂
    h : Eq (Function.comp g₁ preStoneCechUnit) (Function.comp g₂ preStoneCechUnit)
    ⊢ Set.EqOn g₁ g₂ (Set.range preStoneCechUnit)
  -/
  rintro x ⟨x, rfl⟩
  /-
    case intro
    α : Type u
    inst✝² : TopologicalSpace α
    β : Type v
    inst✝¹ : TopologicalSpace β
    inst✝ : T2Space β
    g₁ g₂ : PreStoneCech α → β
    h₁ : Continuous g₁
    h₂ : Continuous g₂
    h : Eq (Function.comp g₁ preStoneCechUnit) (Function.comp g₂ preStoneCechUnit)
    x : α
    ⊢ Eq (g₁ (preStoneCechUnit x)) (g₂ (preStoneCechUnit x))
  -/
  apply congr_fun h x
  /-
    🎉 no goals
  -/


lemma preStoneCechCompat {F G : Ultrafilter α} {x : α} (hF : ↑F ≤ 𝓝 x) (hG : ↑G ≤ 𝓝 x) :
    Ultrafilter.extend g F = Ultrafilter.extend g G := by
  /-
    α : Type u
    inst✝³ : TopologicalSpace α
    β : Type v
    inst✝² : TopologicalSpace β
    inst✝¹ : T2Space β
    inst✝ : CompactSpace β
    g : α → β
    hg : Continuous g
    F G : Ultrafilter α
    x : α
    hF : LE.le (↑F) (nhds x)
    hG : LE.le (↑G) (nhds x)
    ⊢ Eq (Ultrafilter.extend g F) (Ultrafilter.extend g G)
  -/
  replace hF := (map_mono hF).trans hg.continuousAt
  /-
    α : Type u
    inst✝³ : TopologicalSpace α
    β : Type v
    inst✝² : TopologicalSpace β
    inst✝¹ : T2Space β
    inst✝ : CompactSpace β
    g : α → β
    hg : Continuous g
    F G : Ultrafilter α
    x : α
    hG : LE.le (↑G) (nhds x)
    hF : LE.le (Filter.map g ↑F) (nhds (g x))
    ⊢ Eq (Ultrafilter.extend g F) (Ultrafilter.extend g G)
  -/
  replace hG := (map_mono hG).trans hg.continuousAt
  rwa [show Ultrafilter.extend g G = g x by rwa [ultrafilter_extend_eq_iff, G.coe_map],
       ultrafilter_extend_eq_iff, F.coe_map]


/-- The extension of a continuous function from `α` to a compact
  Hausdorff space `β` to the pre-Stone-Čech compactification of `α`. -/
def preStoneCechExtend : PreStoneCech α → β :=
  Quot.lift (Ultrafilter.extend g) fun _ _ ⟨_, hF, hG⟩ ↦ preStoneCechCompat hg hF hG


theorem preStoneCechExtend_extends : preStoneCechExtend hg ∘ preStoneCechUnit = g :=
  ultrafilter_extend_extends g


lemma eq_if_preStoneCechUnit_eq {a b : α} (h : preStoneCechUnit a = preStoneCechUnit b) :
    g a = g b := by
  /-
    α : Type u
    inst✝³ : TopologicalSpace α
    β : Type v
    inst✝² : TopologicalSpace β
    inst✝¹ : T2Space β
    inst✝ : CompactSpace β
    g : α → β
    hg : Continuous g
    a b : α
    h : Eq (preStoneCechUnit a) (preStoneCechUnit b)
    ⊢ Eq (g a) (g b)
  -/
  have e := ultrafilter_extend_extends g
  /-
    α : Type u
    inst✝³ : TopologicalSpace α
    β : Type v
    inst✝² : TopologicalSpace β
    inst✝¹ : T2Space β
    inst✝ : CompactSpace β
    g : α → β
    hg : Continuous g
    a b : α
    h : Eq (preStoneCechUnit a) (preStoneCechUnit b)
    e : Eq (Function.comp (Ultrafilter.extend g) Pure.pure) g
    ⊢ Eq (g a) (g b)
  -/
  rw [← congrFun e a, ← congrFun e b, Function.comp_apply, Function.comp_apply]
  /-
    α : Type u
    inst✝³ : TopologicalSpace α
    β : Type v
    inst✝² : TopologicalSpace β
    inst✝¹ : T2Space β
    inst✝ : CompactSpace β
    g : α → β
    hg : Continuous g
    a b : α
    h : Eq (preStoneCechUnit a) (preStoneCechUnit b)
    e : Eq (Function.comp (Ultrafilter.extend g) Pure.pure) g
    ⊢ Eq (Ultrafilter.extend g (Pure.pure a)) (Ultrafilter.extend g (Pure.pure b))
  -/
  rw [preStoneCechUnit, preStoneCechUnit, Quot.eq] at h
  /-
    α : Type u
    inst✝³ : TopologicalSpace α
    β : Type v
    inst✝² : TopologicalSpace β
    inst✝¹ : T2Space β
    inst✝ : CompactSpace β
    g : α → β
    hg : Continuous g
    a b : α
    h : Relation.EqvGen (fun F G => Exists fun x => And (LE.le (↑F) (nhds x)) (LE. …
    e : Eq (Function.comp (Ultrafilter.extend g) Pure.pure) g
    ⊢ Eq (Ultrafilter.extend g (Pure.pure a)) (Ultrafilter.extend g (Pure.pure b))
  -/
  generalize (pure a : Ultrafilter α) = F at h
  /-
    α : Type u
    inst✝³ : TopologicalSpace α
    β : Type v
    inst✝² : TopologicalSpace β
    inst✝¹ : T2Space β
    inst✝ : CompactSpace β
    g : α → β
    hg : Continuous g
    a b : α
    e : Eq (Function.comp (Ultrafilter.extend g) Pure.pure) g
    F : Ultrafilter α
    h : Relation.EqvGen (fun F G => Exists fun x => And (LE.le (↑F) (nhds x)) (LE. …
    ⊢ Eq (Ultrafilter.extend g F) (Ultrafilter.extend g (Pure.pure b))
  -/
  generalize (pure b : Ultrafilter α) = G at h
  induction h with
  | rel x y a => exact let ⟨a, hx, hy⟩ := a; preStoneCechCompat hg hx hy
  | refl x => rfl
  | symm x y _ h => rw [h]
  | trans x y z _ _ h h' => exact h.trans h'


theorem continuous_preStoneCechExtend : Continuous (preStoneCechExtend hg) :=
  continuous_quot_lift _ (continuous_ultrafilter_extend g)


/-- The Stone-Čech compactification of a topological space. -/
def StoneCech : Type u :=
  t2Quotient (PreStoneCech α)


instance : TopologicalSpace (StoneCech α) :=
  inferInstanceAs <| TopologicalSpace <| t2Quotient _


instance : T2Space (StoneCech α) :=
  inferInstanceAs <| T2Space <| t2Quotient _


instance : CompactSpace (StoneCech α) :=
  Quot.compactSpace


instance [Inhabited α] : Inhabited (StoneCech α) :=
  inferInstanceAs <| Inhabited <| Quotient _


/-- The natural map from α to its Stone-Čech compactification. -/
def stoneCechUnit (x : α) : StoneCech α :=
  t2Quotient.mk (preStoneCechUnit x)


theorem continuous_stoneCechUnit : Continuous (stoneCechUnit : α → StoneCech α) :=
  (t2Quotient.continuous_mk _).comp continuous_preStoneCechUnit


/-- The image of `stoneCechUnit` is dense. (But `stoneCechUnit` need
  not be an embedding, for example if the original space is not Hausdorff.) -/
theorem denseRange_stoneCechUnit : DenseRange (stoneCechUnit : α → StoneCech α) := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    ⊢ DenseRange stoneCechUnit
  -/
  unfold stoneCechUnit t2Quotient.mk
  have : Function.Surjective (t2Quotient.mk : PreStoneCech α → StoneCech α) := by
    exact Quot.mk_surjective
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    this : Function.Surjective t2Quotient.mk
    ⊢ DenseRange fun x => Quotient.mk (t2Setoid (PreStoneCech α)) (preStoneCechUni …
  -/
  exact this.denseRange.comp denseRange_preStoneCechUnit continuous_coinduced_rng
  /-
    🎉 no goals
  -/


theorem stoneCech_hom_ext {g₁ g₂ : StoneCech α → β} (h₁ : Continuous g₁) (h₂ : Continuous g₂)
    (h : g₁ ∘ stoneCechUnit = g₂ ∘ stoneCechUnit) : g₁ = g₂ := by
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    β : Type v
    inst✝¹ : TopologicalSpace β
    inst✝ : T2Space β
    g₁ g₂ : StoneCech α → β
    h₁ : Continuous g₁
    h₂ : Continuous g₂
    h : Eq (Function.comp g₁ stoneCechUnit) (Function.comp g₂ stoneCechUnit)
    ⊢ Eq g₁ g₂
  -/
  apply h₁.ext_on denseRange_stoneCechUnit h₂
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    β : Type v
    inst✝¹ : TopologicalSpace β
    inst✝ : T2Space β
    g₁ g₂ : StoneCech α → β
    h₁ : Continuous g₁
    h₂ : Continuous g₂
    h : Eq (Function.comp g₁ stoneCechUnit) (Function.comp g₂ stoneCechUnit)
    ⊢ Set.EqOn g₁ g₂ (Set.range stoneCechUnit)
  -/
  rintro _ ⟨x, rfl⟩
  /-
    case intro
    α : Type u
    inst✝² : TopologicalSpace α
    β : Type v
    inst✝¹ : TopologicalSpace β
    inst✝ : T2Space β
    g₁ g₂ : StoneCech α → β
    h₁ : Continuous g₁
    h₂ : Continuous g₂
    h : Eq (Function.comp g₁ stoneCechUnit) (Function.comp g₂ stoneCechUnit)
    x : α
    ⊢ Eq (g₁ (stoneCechUnit x)) (g₂ (stoneCechUnit x))
  -/
  exact congr_fun h x
  /-
    🎉 no goals
  -/


/-- The extension of a continuous function from `α` to a compact
  Hausdorff space `β` to the Stone-Čech compactification of `α`.
  This extension implements the universal property of this compactification. -/
def stoneCechExtend : StoneCech α → β :=
  t2Quotient.lift (continuous_preStoneCechExtend hg)


theorem stoneCechExtend_extends : stoneCechExtend hg ∘ stoneCechUnit = g := by
  /-
    α : Type u
    inst✝³ : TopologicalSpace α
    β : Type v
    inst✝² : TopologicalSpace β
    inst✝¹ : T2Space β
    g : α → β
    hg : Continuous g
    inst✝ : CompactSpace β
    ⊢ Eq (Function.comp (stoneCechExtend hg) stoneCechUnit) g
  -/
  ext x
  /-
    case h
    α : Type u
    inst✝³ : TopologicalSpace α
    β : Type v
    inst✝² : TopologicalSpace β
    inst✝¹ : T2Space β
    g : α → β
    hg : Continuous g
    inst✝ : CompactSpace β
    x : α
    ⊢ Eq (Function.comp (stoneCechExtend hg) stoneCechUnit x) (g x)
  -/
  rw [stoneCechExtend, Function.comp_apply, stoneCechUnit, t2Quotient.lift_mk]
  /-
    case h
    α : Type u
    inst✝³ : TopologicalSpace α
    β : Type v
    inst✝² : TopologicalSpace β
    inst✝¹ : T2Space β
    g : α → β
    hg : Continuous g
    inst✝ : CompactSpace β
    x : α
    ⊢ Eq (preStoneCechExtend hg (preStoneCechUnit x)) (g x)
  -/
  apply congrFun (preStoneCechExtend_extends hg)
  /-
    🎉 no goals
  -/


theorem continuous_stoneCechExtend : Continuous (stoneCechExtend hg) :=
  continuous_coinduced_dom.mpr (continuous_preStoneCechExtend hg)


lemma eq_if_stoneCechUnit_eq {a b : α} {f : α → β} (hcf : Continuous f)
    (h : stoneCechUnit a = stoneCechUnit b) : f a = f b := by
  /-
    α : Type u
    inst✝³ : TopologicalSpace α
    β : Type v
    inst✝² : TopologicalSpace β
    inst✝¹ : T2Space β
    inst✝ : CompactSpace β
    a b : α
    f : α → β
    hcf : Continuous f
    h : Eq (stoneCechUnit a) (stoneCechUnit b)
    ⊢ Eq (f a) (f b)
  -/
  rw [← congrFun (stoneCechExtend_extends hcf), ← congrFun (stoneCechExtend_extends hcf)]
  /-
    α : Type u
    inst✝³ : TopologicalSpace α
    β : Type v
    inst✝² : TopologicalSpace β
    inst✝¹ : T2Space β
    inst✝ : CompactSpace β
    a b : α
    f : α → β
    hcf : Continuous f
    h : Eq (stoneCechUnit a) (stoneCechUnit b)
    ⊢ Eq (Function.comp (stoneCechExtend hcf) stoneCechUnit a) (Function.comp (sto …
  -/
  exact congrArg (stoneCechExtend hcf) h
  /-
    🎉 no goals
  -/



theorem CauSeq.tendsto_limit [NormedRing β] [hn : IsAbsoluteValue (norm : β → ℝ)]
    (f : CauSeq β norm) [CauSeq.IsComplete β norm] : Tendsto f atTop (𝓝 f.lim) :=
  tendsto_nhds.mpr
    (by
      /-
        β : Type v
        inst✝¹ : NormedRing β
        hn : IsAbsoluteValue Norm.norm
        f : CauSeq β Norm.norm
        inst✝ : CauSeq.IsComplete β Norm.norm
        ⊢ ∀ (s : Set β), IsOpen s → Membership.mem s f.lim → Membership.mem Filter.atT …
      -/
      intro s os lfs
      /-
        β : Type v
        inst✝¹ : NormedRing β
        hn : IsAbsoluteValue Norm.norm
        f : CauSeq β Norm.norm
        inst✝ : CauSeq.IsComplete β Norm.norm
        s : Set β
        os : IsOpen s
        lfs : Membership.mem s f.lim
        ⊢ Membership.mem Filter.atTop (Set.preimage (↑f) s)
      -/
      suffices ∃ a : ℕ, ∀ b : ℕ, b ≥ a → f b ∈ s by simpa using this
      /-
        β : Type v
        inst✝¹ : NormedRing β
        hn : IsAbsoluteValue Norm.norm
        f : CauSeq β Norm.norm
        inst✝ : CauSeq.IsComplete β Norm.norm
        s : Set β
        os : IsOpen s
        lfs : Membership.mem s f.lim
        ⊢ Exists fun a => ∀ (b : Nat), GE.ge b a → Membership.mem s (↑f b)
      -/
      rcases Metric.isOpen_iff.1 os _ lfs with ⟨ε, ⟨hε, hεs⟩⟩
      /-
        case intro.intro
        β : Type v
        inst✝¹ : NormedRing β
        hn : IsAbsoluteValue Norm.norm
        f : CauSeq β Norm.norm
        inst✝ : CauSeq.IsComplete β Norm.norm
        s : Set β
        os : IsOpen s
        lfs : Membership.mem s f.lim
        ε : Real
        hε : GT.gt ε 0
        hεs : HasSubset.Subset (Metric.ball f.lim ε) s
        ⊢ Exists fun a => ∀ (b : Nat), GE.ge b a → Membership.mem s (↑f b)
      -/
      cases' Setoid.symm (CauSeq.equiv_lim f) _ hε with N hN
      /-
        case intro.intro.intro
        β : Type v
        inst✝¹ : NormedRing β
        hn : IsAbsoluteValue Norm.norm
        f : CauSeq β Norm.norm
        inst✝ : CauSeq.IsComplete β Norm.norm
        s : Set β
        os : IsOpen s
        lfs : Membership.mem s f.lim
        ε : Real
        hε : GT.gt ε 0
        hεs : HasSubset.Subset (Metric.ball f.lim ε) s
        N : Nat
        hN : ∀ (j : Nat), GE.ge j N → LT.lt (Norm.norm (↑(HSub.hSub (CauSeq.const Norm …
        ⊢ Exists fun a => ∀ (b : Nat), GE.ge b a → Membership.mem s (↑f b)
      -/
      exists N
      /-
        case intro.intro.intro
        β : Type v
        inst✝¹ : NormedRing β
        hn : IsAbsoluteValue Norm.norm
        f : CauSeq β Norm.norm
        inst✝ : CauSeq.IsComplete β Norm.norm
        s : Set β
        os : IsOpen s
        lfs : Membership.mem s f.lim
        ε : Real
        hε : GT.gt ε 0
        hεs : HasSubset.Subset (Metric.ball f.lim ε) s
        N : Nat
        hN : ∀ (j : Nat), GE.ge j N → LT.lt (Norm.norm (↑(HSub.hSub (CauSeq.const Norm …
        ⊢ ∀ (b : Nat), GE.ge b N → Membership.mem s (↑f b)
      -/
      intro b hb
      /-
        case intro.intro.intro
        β : Type v
        inst✝¹ : NormedRing β
        hn : IsAbsoluteValue Norm.norm
        f : CauSeq β Norm.norm
        inst✝ : CauSeq.IsComplete β Norm.norm
        s : Set β
        os : IsOpen s
        lfs : Membership.mem s f.lim
        ε : Real
        hε : GT.gt ε 0
        hεs : HasSubset.Subset (Metric.ball f.lim ε) s
        N : Nat
        hN : ∀ (j : Nat), GE.ge j N → LT.lt (Norm.norm (↑(HSub.hSub (CauSeq.const Norm …
        b : Nat
        hb : GE.ge b N
        ⊢ Membership.mem s (↑f b)
      -/
      apply hεs
      /-
        case intro.intro.intro.a
        β : Type v
        inst✝¹ : NormedRing β
        hn : IsAbsoluteValue Norm.norm
        f : CauSeq β Norm.norm
        inst✝ : CauSeq.IsComplete β Norm.norm
        s : Set β
        os : IsOpen s
        lfs : Membership.mem s f.lim
        ε : Real
        hε : GT.gt ε 0
        hεs : HasSubset.Subset (Metric.ball f.lim ε) s
        N : Nat
        hN : ∀ (j : Nat), GE.ge j N → LT.lt (Norm.norm (↑(HSub.hSub (CauSeq.const Norm …
        b : Nat
        hb : GE.ge b N
        ⊢ Membership.mem (Metric.ball f.lim ε) (↑f b)
      -/
      dsimp [Metric.ball]
      /-
        case intro.intro.intro.a
        β : Type v
        inst✝¹ : NormedRing β
        hn : IsAbsoluteValue Norm.norm
        f : CauSeq β Norm.norm
        inst✝ : CauSeq.IsComplete β Norm.norm
        s : Set β
        os : IsOpen s
        lfs : Membership.mem s f.lim
        ε : Real
        hε : GT.gt ε 0
        hεs : HasSubset.Subset (Metric.ball f.lim ε) s
        N : Nat
        hN : ∀ (j : Nat), GE.ge j N → LT.lt (Norm.norm (↑(HSub.hSub (CauSeq.const Norm …
        b : Nat
        hb : GE.ge b N
        ⊢ LT.lt (Dist.dist (↑f b) f.lim) ε
      -/
      rw [dist_comm, dist_eq_norm]
      /-
        case intro.intro.intro.a
        β : Type v
        inst✝¹ : NormedRing β
        hn : IsAbsoluteValue Norm.norm
        f : CauSeq β Norm.norm
        inst✝ : CauSeq.IsComplete β Norm.norm
        s : Set β
        os : IsOpen s
        lfs : Membership.mem s f.lim
        ε : Real
        hε : GT.gt ε 0
        hεs : HasSubset.Subset (Metric.ball f.lim ε) s
        N : Nat
        hN : ∀ (j : Nat), GE.ge j N → LT.lt (Norm.norm (↑(HSub.hSub (CauSeq.const Norm …
        b : Nat
        hb : GE.ge b N
        ⊢ LT.lt (Norm.norm (HSub.hSub f.lim (↑f b))) ε
      -/
      solve_by_elim)
      /-
        🎉 no goals
      -/


theorem CauchySeq.isCauSeq {f : ℕ → β} (hf : CauchySeq f) : IsCauSeq norm f := by
  /-
    β : Type v
    inst✝ : NormedField β
    f : Nat → β
    hf : CauchySeq f
    ⊢ IsCauSeq Norm.norm f
  -/
  cases' cauchy_iff.1 hf with hf1 hf2
  /-
    case intro
    β : Type v
    inst✝ : NormedField β
    f : Nat → β
    hf : CauchySeq f
    hf1 : (Filter.map f Filter.atTop).NeBot
    hf2 : ∀ (s : Set (Prod β β)), Membership.mem (uniformity β) s → Exists fun t = …
    ⊢ IsCauSeq Norm.norm f
  -/
  intro ε hε
  /-
    case intro
    β : Type v
    inst✝ : NormedField β
    f : Nat → β
    hf : CauchySeq f
    hf1 : (Filter.map f Filter.atTop).NeBot
    hf2 : ∀ (s : Set (Prod β β)), Membership.mem (uniformity β) s → Exists fun t = …
    ε : Real
    hε : GT.gt ε 0
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (Norm.norm (HSub.hSub (f j) ( …
  -/
  rcases hf2 { x | dist x.1 x.2 < ε } (dist_mem_uniformity hε) with ⟨t, ⟨ht, htsub⟩⟩
  /-
    case intro.intro.intro
    β : Type v
    inst✝ : NormedField β
    f : Nat → β
    hf : CauchySeq f
    hf1 : (Filter.map f Filter.atTop).NeBot
    hf2 : ∀ (s : Set (Prod β β)), Membership.mem (uniformity β) s → Exists fun t = …
    ε : Real
    hε : GT.gt ε 0
    t : Set β
    ht : Membership.mem (Filter.map f Filter.atTop) t
    htsub : HasSubset.Subset (SProd.sprod t t) (setOf fun x => LT.lt (Dist.dist x. …
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (Norm.norm (HSub.hSub (f j) ( …
  -/
  simp only [mem_map, mem_atTop_sets, mem_preimage] at ht; cases' ht with N hN
  /-
    case intro.intro.intro.intro
    β : Type v
    inst✝ : NormedField β
    f : Nat → β
    hf : CauchySeq f
    hf1 : (Filter.map f Filter.atTop).NeBot
    hf2 : ∀ (s : Set (Prod β β)), Membership.mem (uniformity β) s → Exists fun t = …
    ε : Real
    hε : GT.gt ε 0
    t : Set β
    htsub : HasSubset.Subset (SProd.sprod t t) (setOf fun x => LT.lt (Dist.dist x. …
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → Membership.mem t (f b)
    ⊢ Exists fun i => ∀ (j : Nat), GE.ge j i → LT.lt (Norm.norm (HSub.hSub (f j) ( …
  -/
  exists N
  /-
    case intro.intro.intro.intro
    β : Type v
    inst✝ : NormedField β
    f : Nat → β
    hf : CauchySeq f
    hf1 : (Filter.map f Filter.atTop).NeBot
    hf2 : ∀ (s : Set (Prod β β)), Membership.mem (uniformity β) s → Exists fun t = …
    ε : Real
    hε : GT.gt ε 0
    t : Set β
    htsub : HasSubset.Subset (SProd.sprod t t) (setOf fun x => LT.lt (Dist.dist x. …
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → Membership.mem t (f b)
    ⊢ ∀ (j : Nat), GE.ge j N → LT.lt (Norm.norm (HSub.hSub (f j) (f N))) ε
  -/
  intro j hj
  /-
    case intro.intro.intro.intro
    β : Type v
    inst✝ : NormedField β
    f : Nat → β
    hf : CauchySeq f
    hf1 : (Filter.map f Filter.atTop).NeBot
    hf2 : ∀ (s : Set (Prod β β)), Membership.mem (uniformity β) s → Exists fun t = …
    ε : Real
    hε : GT.gt ε 0
    t : Set β
    htsub : HasSubset.Subset (SProd.sprod t t) (setOf fun x => LT.lt (Dist.dist x. …
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → Membership.mem t (f b)
    j : Nat
    hj : GE.ge j N
    ⊢ LT.lt (Norm.norm (HSub.hSub (f j) (f N))) ε
  -/
  rw [← dist_eq_norm]
  /-
    case intro.intro.intro.intro
    β : Type v
    inst✝ : NormedField β
    f : Nat → β
    hf : CauchySeq f
    hf1 : (Filter.map f Filter.atTop).NeBot
    hf2 : ∀ (s : Set (Prod β β)), Membership.mem (uniformity β) s → Exists fun t = …
    ε : Real
    hε : GT.gt ε 0
    t : Set β
    htsub : HasSubset.Subset (SProd.sprod t t) (setOf fun x => LT.lt (Dist.dist x. …
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → Membership.mem t (f b)
    j : Nat
    hj : GE.ge j N
    ⊢ LT.lt (Dist.dist (f j) (f N)) ε
  -/
  apply @htsub (f j, f N)
  /-
    case intro.intro.intro.intro
    β : Type v
    inst✝ : NormedField β
    f : Nat → β
    hf : CauchySeq f
    hf1 : (Filter.map f Filter.atTop).NeBot
    hf2 : ∀ (s : Set (Prod β β)), Membership.mem (uniformity β) s → Exists fun t = …
    ε : Real
    hε : GT.gt ε 0
    t : Set β
    htsub : HasSubset.Subset (SProd.sprod t t) (setOf fun x => LT.lt (Dist.dist x. …
    N : Nat
    hN : ∀ (b : Nat), GE.ge b N → Membership.mem t (f b)
    j : Nat
    hj : GE.ge j N
    ⊢ Membership.mem (SProd.sprod t t) { fst := f j, snd := f N }
  -/
                            /-
                              🎉 no goals
                            -/
  apply Set.mk_mem_prod <;> solve_by_elim [le_refl]
                            /-
                              🎉 no goals
                            -/


theorem CauSeq.cauchySeq (f : CauSeq β norm) : CauchySeq f := by
  /-
    β : Type v
    inst✝ : NormedField β
    f : CauSeq β Norm.norm
    ⊢ CauchySeq ↑f
  -/
  refine cauchy_iff.2 ⟨by infer_instance, fun s hs => ?_⟩
  /-
    β : Type v
    inst✝ : NormedField β
    f : CauSeq β Norm.norm
    s : Set (Prod β β)
    hs : Membership.mem (uniformity β) s
    ⊢ Exists fun t => And (Membership.mem (Filter.map (↑f) Filter.atTop) t) (HasSu …
  -/
  rcases mem_uniformity_dist.1 hs with ⟨ε, ⟨hε, hεs⟩⟩
  /-
    case intro.intro
    β : Type v
    inst✝ : NormedField β
    f : CauSeq β Norm.norm
    s : Set (Prod β β)
    hs : Membership.mem (uniformity β) s
    ε : Real
    hε : GT.gt ε 0
    hεs : ∀ ⦃a b : β⦄, LT.lt (Dist.dist a b) ε → Membership.mem s { fst := a, snd  …
    ⊢ Exists fun t => And (Membership.mem (Filter.map (↑f) Filter.atTop) t) (HasSu …
  -/
  cases' CauSeq.cauchy₂ f hε with N hN
  /-
    case intro.intro.intro
    β : Type v
    inst✝ : NormedField β
    f : CauSeq β Norm.norm
    s : Set (Prod β β)
    hs : Membership.mem (uniformity β) s
    ε : Real
    hε : GT.gt ε 0
    hεs : ∀ ⦃a b : β⦄, LT.lt (Dist.dist a b) ε → Membership.mem s { fst := a, snd  …
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (Norm.norm (HSub. …
    ⊢ Exists fun t => And (Membership.mem (Filter.map (↑f) Filter.atTop) t) (HasSu …
  -/
  exists { n | n ≥ N }.image f
  /-
    case intro.intro.intro
    β : Type v
    inst✝ : NormedField β
    f : CauSeq β Norm.norm
    s : Set (Prod β β)
    hs : Membership.mem (uniformity β) s
    ε : Real
    hε : GT.gt ε 0
    hεs : ∀ ⦃a b : β⦄, LT.lt (Dist.dist a b) ε → Membership.mem s { fst := a, snd  …
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (Norm.norm (HSub. …
    ⊢ And (Membership.mem (Filter.map (↑f) Filter.atTop) (Set.image (↑f) (setOf fu …
  -/
  simp only [exists_prop, mem_atTop_sets, mem_map, mem_image, mem_setOf_eq]
  /-
    case intro.intro.intro
    β : Type v
    inst✝ : NormedField β
    f : CauSeq β Norm.norm
    s : Set (Prod β β)
    hs : Membership.mem (uniformity β) s
    ε : Real
    hε : GT.gt ε 0
    hεs : ∀ ⦃a b : β⦄, LT.lt (Dist.dist a b) ε → Membership.mem s { fst := a, snd  …
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (Norm.norm (HSub. …
    ⊢ And (Exists fun a => ∀ (b : Nat), GE.ge b a → Membership.mem (Set.preimage ( …
  -/
  constructor
    /-
      case intro.intro.intro.left
      β : Type v
      inst✝ : NormedField β
      f : CauSeq β Norm.norm
      s : Set (Prod β β)
      hs : Membership.mem (uniformity β) s
      ε : Real
      hε : GT.gt ε 0
      hεs : ∀ ⦃a b : β⦄, LT.lt (Dist.dist a b) ε → Membership.mem s { fst := a, snd  …
      N : Nat
      hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (Norm.norm (HSub. …
      ⊢ Exists fun a => ∀ (b : Nat), GE.ge b a → Membership.mem (Set.preimage (↑f) ( …
    -/
  · exists N
    /-
      case intro.intro.intro.left
      β : Type v
      inst✝ : NormedField β
      f : CauSeq β Norm.norm
      s : Set (Prod β β)
      hs : Membership.mem (uniformity β) s
      ε : Real
      hε : GT.gt ε 0
      hεs : ∀ ⦃a b : β⦄, LT.lt (Dist.dist a b) ε → Membership.mem s { fst := a, snd  …
      N : Nat
      hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (Norm.norm (HSub. …
      ⊢ ∀ (b : Nat), GE.ge b N → Membership.mem (Set.preimage (↑f) (Set.image (↑f) ( …
    -/
    intro b hb
    /-
      case intro.intro.intro.left
      β : Type v
      inst✝ : NormedField β
      f : CauSeq β Norm.norm
      s : Set (Prod β β)
      hs : Membership.mem (uniformity β) s
      ε : Real
      hε : GT.gt ε 0
      hεs : ∀ ⦃a b : β⦄, LT.lt (Dist.dist a b) ε → Membership.mem s { fst := a, snd  …
      N : Nat
      hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (Norm.norm (HSub. …
      b : Nat
      hb : GE.ge b N
      ⊢ Membership.mem (Set.preimage (↑f) (Set.image (↑f) (setOf fun n => GE.ge n N) …
    -/
    exists b
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.right
      β : Type v
      inst✝ : NormedField β
      f : CauSeq β Norm.norm
      s : Set (Prod β β)
      hs : Membership.mem (uniformity β) s
      ε : Real
      hε : GT.gt ε 0
      hεs : ∀ ⦃a b : β⦄, LT.lt (Dist.dist a b) ε → Membership.mem s { fst := a, snd  …
      N : Nat
      hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (Norm.norm (HSub. …
      ⊢ HasSubset.Subset (SProd.sprod (Set.image (↑f) (setOf fun n => GE.ge n N)) (S …
    -/
  · rintro ⟨a, b⟩ ⟨⟨a', ⟨ha'1, ha'2⟩⟩, ⟨b', ⟨hb'1, hb'2⟩⟩⟩
    /-
      case intro.intro.intro.right.mk.intro.intro.intro.intro.intro
      β : Type v
      inst✝ : NormedField β
      f : CauSeq β Norm.norm
      s : Set (Prod β β)
      hs : Membership.mem (uniformity β) s
      ε : Real
      hε : GT.gt ε 0
      hεs : ∀ ⦃a b : β⦄, LT.lt (Dist.dist a b) ε → Membership.mem s { fst := a, snd  …
      N : Nat
      hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (Norm.norm (HSub. …
      a b : β
      a' : Nat
      ha'1 : Membership.mem (setOf fun n => GE.ge n N) a'
      ha'2 : Eq (↑f a') { fst := a, snd := b }.1
      b' : Nat
      hb'1 : Membership.mem (setOf fun n => GE.ge n N) b'
      hb'2 : Eq (↑f b') { fst := a, snd := b }.2
      ⊢ Membership.mem s { fst := a, snd := b }
    -/
    dsimp at ha'1 ha'2 hb'1 hb'2
    /-
      case intro.intro.intro.right.mk.intro.intro.intro.intro.intro
      β : Type v
      inst✝ : NormedField β
      f : CauSeq β Norm.norm
      s : Set (Prod β β)
      hs : Membership.mem (uniformity β) s
      ε : Real
      hε : GT.gt ε 0
      hεs : ∀ ⦃a b : β⦄, LT.lt (Dist.dist a b) ε → Membership.mem s { fst := a, snd  …
      N : Nat
      hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (Norm.norm (HSub. …
      a b : β
      a' : Nat
      ha'1 : GE.ge a' N
      ha'2 : Eq (↑f a') a
      b' : Nat
      hb'1 : GE.ge b' N
      hb'2 : Eq (↑f b') b
      ⊢ Membership.mem s { fst := a, snd := b }
    -/
    rw [← ha'2, ← hb'2]
    /-
      case intro.intro.intro.right.mk.intro.intro.intro.intro.intro
      β : Type v
      inst✝ : NormedField β
      f : CauSeq β Norm.norm
      s : Set (Prod β β)
      hs : Membership.mem (uniformity β) s
      ε : Real
      hε : GT.gt ε 0
      hεs : ∀ ⦃a b : β⦄, LT.lt (Dist.dist a b) ε → Membership.mem s { fst := a, snd  …
      N : Nat
      hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (Norm.norm (HSub. …
      a b : β
      a' : Nat
      ha'1 : GE.ge a' N
      ha'2 : Eq (↑f a') a
      b' : Nat
      hb'1 : GE.ge b' N
      hb'2 : Eq (↑f b') b
      ⊢ Membership.mem s { fst := ↑f a', snd := ↑f b' }
    -/
    apply hεs
    /-
      case intro.intro.intro.right.mk.intro.intro.intro.intro.intro.a
      β : Type v
      inst✝ : NormedField β
      f : CauSeq β Norm.norm
      s : Set (Prod β β)
      hs : Membership.mem (uniformity β) s
      ε : Real
      hε : GT.gt ε 0
      hεs : ∀ ⦃a b : β⦄, LT.lt (Dist.dist a b) ε → Membership.mem s { fst := a, snd  …
      N : Nat
      hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (Norm.norm (HSub. …
      a b : β
      a' : Nat
      ha'1 : GE.ge a' N
      ha'2 : Eq (↑f a') a
      b' : Nat
      hb'1 : GE.ge b' N
      hb'2 : Eq (↑f b') b
      ⊢ LT.lt (Dist.dist (↑f a') (↑f b')) ε
    -/
    rw [dist_eq_norm]
    /-
      case intro.intro.intro.right.mk.intro.intro.intro.intro.intro.a
      β : Type v
      inst✝ : NormedField β
      f : CauSeq β Norm.norm
      s : Set (Prod β β)
      hs : Membership.mem (uniformity β) s
      ε : Real
      hε : GT.gt ε 0
      hεs : ∀ ⦃a b : β⦄, LT.lt (Dist.dist a b) ε → Membership.mem s { fst := a, snd  …
      N : Nat
      hN : ∀ (j : Nat), GE.ge j N → ∀ (k : Nat), GE.ge k N → LT.lt (Norm.norm (HSub. …
      a b : β
      a' : Nat
      ha'1 : GE.ge a' N
      ha'2 : Eq (↑f a') a
      b' : Nat
      hb'1 : GE.ge b' N
      hb'2 : Eq (↑f b') b
      ⊢ LT.lt (Norm.norm (HSub.hSub (↑f a') (↑f b'))) ε
    -/
                 /-
                   🎉 no goals
                 -/
    apply hN <;> assumption
                 /-
                   🎉 no goals
                 -/


/-- In a normed field, `CauSeq` coincides with the usual notion of Cauchy sequences. -/
theorem isCauSeq_iff_cauchySeq {α : Type u} [NormedField α] {u : ℕ → α} :
    IsCauSeq norm u ↔ CauchySeq u :=
  ⟨fun h => CauSeq.cauchySeq ⟨u, h⟩, fun h => h.isCauSeq⟩

-- see Note [lower instance priority]

/-- A complete normed field is complete as a metric space, as Cauchy sequences converge by
assumption and this suffices to characterize completeness. -/
instance (priority := 100) completeSpace_of_cauSeq_isComplete [CauSeq.IsComplete β norm] :
    CompleteSpace β := by
  /-
    β : Type v
    inst✝¹ : NormedField β
    inst✝ : CauSeq.IsComplete β Norm.norm
    ⊢ CompleteSpace β
  -/
  apply complete_of_cauchySeq_tendsto
  /-
    case a
    β : Type v
    inst✝¹ : NormedField β
    inst✝ : CauSeq.IsComplete β Norm.norm
    ⊢ ∀ (u : Nat → β), CauchySeq u → Exists fun a => Filter.Tendsto u Filter.atTop …
  -/
  intro u hu
  /-
    case a
    β : Type v
    inst✝¹ : NormedField β
    inst✝ : CauSeq.IsComplete β Norm.norm
    u : Nat → β
    hu : CauchySeq u
    ⊢ Exists fun a => Filter.Tendsto u Filter.atTop (nhds a)
  -/
  have C : IsCauSeq norm u := isCauSeq_iff_cauchySeq.2 hu
  /-
    case a
    β : Type v
    inst✝¹ : NormedField β
    inst✝ : CauSeq.IsComplete β Norm.norm
    u : Nat → β
    hu : CauchySeq u
    C : IsCauSeq Norm.norm u
    ⊢ Exists fun a => Filter.Tendsto u Filter.atTop (nhds a)
  -/
  exists CauSeq.lim ⟨u, C⟩
  /-
    case a
    β : Type v
    inst✝¹ : NormedField β
    inst✝ : CauSeq.IsComplete β Norm.norm
    u : Nat → β
    hu : CauchySeq u
    C : IsCauSeq Norm.norm u
    ⊢ Filter.Tendsto u Filter.atTop (nhds (CauSeq.lim ⟨u, C⟩))
  -/
  rw [Metric.tendsto_atTop]
  /-
    case a
    β : Type v
    inst✝¹ : NormedField β
    inst✝ : CauSeq.IsComplete β Norm.norm
    u : Nat → β
    hu : CauchySeq u
    C : IsCauSeq Norm.norm u
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt (Di …
  -/
  intro ε εpos
  /-
    case a
    β : Type v
    inst✝¹ : NormedField β
    inst✝ : CauSeq.IsComplete β Norm.norm
    u : Nat → β
    hu : CauchySeq u
    C : IsCauSeq Norm.norm u
    ε : Real
    εpos : GT.gt ε 0
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist (u n) (CauSeq.lim  …
  -/
  cases' (CauSeq.equiv_lim ⟨u, C⟩) _ εpos with N hN
  /-
    case a.intro
    β : Type v
    inst✝¹ : NormedField β
    inst✝ : CauSeq.IsComplete β Norm.norm
    u : Nat → β
    hu : CauchySeq u
    C : IsCauSeq Norm.norm u
    ε : Real
    εpos : GT.gt ε 0
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → LT.lt (Norm.norm (↑(HSub.hSub ⟨u, C⟩ (CauSeq.con …
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist (u n) (CauSeq.lim  …
  -/
  exists N
  /-
    case a.intro
    β : Type v
    inst✝¹ : NormedField β
    inst✝ : CauSeq.IsComplete β Norm.norm
    u : Nat → β
    hu : CauchySeq u
    C : IsCauSeq Norm.norm u
    ε : Real
    εpos : GT.gt ε 0
    N : Nat
    hN : ∀ (j : Nat), GE.ge j N → LT.lt (Norm.norm (↑(HSub.hSub ⟨u, C⟩ (CauSeq.con …
    ⊢ ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist (u n) (CauSeq.lim ⟨u, C⟩)) ε
  -/
  simpa [dist_eq_norm] using hN
  /-
    🎉 no goals
  -/


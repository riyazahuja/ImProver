/-- A measurable set `s` is a *fundamental domain* for an additive action of an additive group `G`
on a measurable space `α` with respect to a measure `α` if the sets `g +ᵥ s`, `g : G`, are pairwise
a.e. disjoint and cover the whole space. -/
structure IsAddFundamentalDomain (G : Type*) {α : Type*} [Zero G] [VAdd G α] [MeasurableSpace α]
    (s : Set α) (μ : Measure α := by volume_tac) : Prop where
  protected nullMeasurableSet : NullMeasurableSet s μ
  protected ae_covers : ∀ᵐ x ∂μ, ∃ g : G, g +ᵥ x ∈ s
  protected aedisjoint : Pairwise <| (AEDisjoint μ on fun g : G => g +ᵥ s)


/-- A measurable set `s` is a *fundamental domain* for an action of a group `G` on a measurable
space `α` with respect to a measure `α` if the sets `g • s`, `g : G`, are pairwise a.e. disjoint and
cover the whole space. -/
@[to_additive IsAddFundamentalDomain]
structure IsFundamentalDomain (G : Type*) {α : Type*} [One G] [SMul G α] [MeasurableSpace α]
    (s : Set α) (μ : Measure α := by volume_tac) : Prop where
  protected nullMeasurableSet : NullMeasurableSet s μ
  protected ae_covers : ∀ᵐ x ∂μ, ∃ g : G, g • x ∈ s
  protected aedisjoint : Pairwise <| (AEDisjoint μ on fun g : G => g • s)


/-- If for each `x : α`, exactly one of `g • x`, `g : G`, belongs to a measurable set `s`, then `s`
is a fundamental domain for the action of `G` on `α`. -/
@[to_additive "If for each `x : α`, exactly one of `g +ᵥ x`, `g : G`, belongs to a measurable set
`s`, then `s` is a fundamental domain for the additive action of `G` on `α`."]
theorem mk' (h_meas : NullMeasurableSet s μ) (h_exists : ∀ x : α, ∃! g : G, g • x ∈ s) :
    IsFundamentalDomain G s μ where
  nullMeasurableSet := h_meas
  ae_covers := Eventually.of_forall fun x => (h_exists x).exists
  aedisjoint a b hab := Disjoint.aedisjoint <| disjoint_left.2 fun x hxa hxb => by
    /-
      G : Type u_1
      α : Type u_3
      inst✝² : Group G
      inst✝¹ : MulAction G α
      inst✝ : MeasurableSpace α
      s : Set α
      μ : MeasureTheory.Measure α
      h_meas : MeasureTheory.NullMeasurableSet s μ
      h_exists : ∀ (x : α), ExistsUnique fun g => Membership.mem s (HSMul.hSMul g x)
      a b : G
      hab : Ne a b
      x : α
      hxa : Membership.mem ((fun g => HSMul.hSMul g s) a) x
      hxb : Membership.mem ((fun g => HSMul.hSMul g s) b) x
      ⊢ False
    -/
    rw [mem_smul_set_iff_inv_smul_mem] at hxa hxb
    /-
      G : Type u_1
      α : Type u_3
      inst✝² : Group G
      inst✝¹ : MulAction G α
      inst✝ : MeasurableSpace α
      s : Set α
      μ : MeasureTheory.Measure α
      h_meas : MeasureTheory.NullMeasurableSet s μ
      h_exists : ∀ (x : α), ExistsUnique fun g => Membership.mem s (HSMul.hSMul g x)
      a b : G
      hab : Ne a b
      x : α
      hxa : Membership.mem s (HSMul.hSMul (Inv.inv a) x)
      hxb : Membership.mem s (HSMul.hSMul (Inv.inv b) x)
      ⊢ False
    -/
    exact hab (inv_injective <| (h_exists x).unique hxa hxb)
    /-
      🎉 no goals
    -/


/-- For `s` to be a fundamental domain, it's enough to check
`MeasureTheory.AEDisjoint (g • s) s` for `g ≠ 1`. -/
@[to_additive "For `s` to be a fundamental domain, it's enough to check
  `MeasureTheory.AEDisjoint (g +ᵥ s) s` for `g ≠ 0`."]
theorem mk'' (h_meas : NullMeasurableSet s μ) (h_ae_covers : ∀ᵐ x ∂μ, ∃ g : G, g • x ∈ s)
    (h_ae_disjoint : ∀ g, g ≠ (1 : G) → AEDisjoint μ (g • s) s)
    (h_qmp : ∀ g : G, QuasiMeasurePreserving ((g • ·) : α → α) μ μ) :
    IsFundamentalDomain G s μ where
  nullMeasurableSet := h_meas
  ae_covers := h_ae_covers
  aedisjoint := pairwise_aedisjoint_of_aedisjoint_forall_ne_one h_ae_disjoint h_qmp


/-- If a measurable space has a finite measure `μ` and a countable group `G` acts
quasi-measure-preservingly, then to show that a set `s` is a fundamental domain, it is sufficient
to check that its translates `g • s` are (almost) disjoint and that the sum `∑' g, μ (g • s)` is
sufficiently large. -/
@[to_additive
  "If a measurable space has a finite measure `μ` and a countable additive group `G` acts
  quasi-measure-preservingly, then to show that a set `s` is a fundamental domain, it is sufficient
  to check that its translates `g +ᵥ s` are (almost) disjoint and that the sum `∑' g, μ (g +ᵥ s)` is
  sufficiently large."]
theorem mk_of_measure_univ_le [IsFiniteMeasure μ] [Countable G] (h_meas : NullMeasurableSet s μ)
    (h_ae_disjoint : ∀ g ≠ (1 : G), AEDisjoint μ (g • s) s)
    (h_qmp : ∀ g : G, QuasiMeasurePreserving (g • · : α → α) μ μ)
    (h_measure_univ_le : μ (univ : Set α) ≤ ∑' g : G, μ (g • s)) : IsFundamentalDomain G s μ :=
  have aedisjoint : Pairwise (AEDisjoint μ on fun g : G => g • s) :=
    pairwise_aedisjoint_of_aedisjoint_forall_ne_one h_ae_disjoint h_qmp
  { nullMeasurableSet := h_meas
    aedisjoint
    ae_covers := by
      replace h_meas : ∀ g : G, NullMeasurableSet (g • s) μ := fun g => by
        rw [← inv_inv g, ← preimage_smul]; exact h_meas.preimage (h_qmp g⁻¹)
      have h_meas' : NullMeasurableSet {a | ∃ g : G, g • a ∈ s} μ := by
        rw [← iUnion_smul_eq_setOf_exists]; exact .iUnion h_meas
      /-
        G : Type u_1
        α : Type u_3
        inst✝⁴ : Group G
        inst✝³ : MulAction G α
        inst✝² : MeasurableSpace α
        s : Set α
        μ : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.IsFiniteMeasure μ
        inst✝ : Countable G
        h_ae_disjoint : ∀ (g : G), Ne g 1 → MeasureTheory.AEDisjoint μ (HSMul.hSMul g  …
        h_qmp : ∀ (g : G), MeasureTheory.Measure.QuasiMeasurePreserving (fun x => HSMu …
        h_measure_univ_le : LE.le (μ Set.univ) (tsum fun g => μ (HSMul.hSMul g s))
        aedisjoint : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) fun g => HS …
        h_meas : ∀ (g : G), MeasureTheory.NullMeasurableSet (HSMul.hSMul g s) μ
        h_meas' : MeasureTheory.NullMeasurableSet (setOf fun a => Exists fun g => Memb …
        ⊢ Filter.Eventually (fun x => Exists fun g => Membership.mem s (HSMul.hSMul g  …
      -/
      rw [ae_iff_measure_eq h_meas', ← iUnion_smul_eq_setOf_exists]
      /-
        G : Type u_1
        α : Type u_3
        inst✝⁴ : Group G
        inst✝³ : MulAction G α
        inst✝² : MeasurableSpace α
        s : Set α
        μ : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.IsFiniteMeasure μ
        inst✝ : Countable G
        h_ae_disjoint : ∀ (g : G), Ne g 1 → MeasureTheory.AEDisjoint μ (HSMul.hSMul g  …
        h_qmp : ∀ (g : G), MeasureTheory.Measure.QuasiMeasurePreserving (fun x => HSMu …
        h_measure_univ_le : LE.le (μ Set.univ) (tsum fun g => μ (HSMul.hSMul g s))
        aedisjoint : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) fun g => HS …
        h_meas : ∀ (g : G), MeasureTheory.NullMeasurableSet (HSMul.hSMul g s) μ
        h_meas' : MeasureTheory.NullMeasurableSet (setOf fun a => Exists fun g => Memb …
        ⊢ Eq (μ (Set.iUnion fun g => HSMul.hSMul g s)) (μ Set.univ)
      -/
      refine le_antisymm (measure_mono <| subset_univ _) ?_
      /-
        G : Type u_1
        α : Type u_3
        inst✝⁴ : Group G
        inst✝³ : MulAction G α
        inst✝² : MeasurableSpace α
        s : Set α
        μ : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.IsFiniteMeasure μ
        inst✝ : Countable G
        h_ae_disjoint : ∀ (g : G), Ne g 1 → MeasureTheory.AEDisjoint μ (HSMul.hSMul g  …
        h_qmp : ∀ (g : G), MeasureTheory.Measure.QuasiMeasurePreserving (fun x => HSMu …
        h_measure_univ_le : LE.le (μ Set.univ) (tsum fun g => μ (HSMul.hSMul g s))
        aedisjoint : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) fun g => HS …
        h_meas : ∀ (g : G), MeasureTheory.NullMeasurableSet (HSMul.hSMul g s) μ
        h_meas' : MeasureTheory.NullMeasurableSet (setOf fun a => Exists fun g => Memb …
        ⊢ LE.le (μ Set.univ) (μ (Set.iUnion fun g => HSMul.hSMul g s))
      -/
      rw [measure_iUnion₀ aedisjoint h_meas]
      /-
        G : Type u_1
        α : Type u_3
        inst✝⁴ : Group G
        inst✝³ : MulAction G α
        inst✝² : MeasurableSpace α
        s : Set α
        μ : MeasureTheory.Measure α
        inst✝¹ : MeasureTheory.IsFiniteMeasure μ
        inst✝ : Countable G
        h_ae_disjoint : ∀ (g : G), Ne g 1 → MeasureTheory.AEDisjoint μ (HSMul.hSMul g  …
        h_qmp : ∀ (g : G), MeasureTheory.Measure.QuasiMeasurePreserving (fun x => HSMu …
        h_measure_univ_le : LE.le (μ Set.univ) (tsum fun g => μ (HSMul.hSMul g s))
        aedisjoint : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) fun g => HS …
        h_meas : ∀ (g : G), MeasureTheory.NullMeasurableSet (HSMul.hSMul g s) μ
        h_meas' : MeasureTheory.NullMeasurableSet (setOf fun a => Exists fun g => Memb …
        ⊢ LE.le (μ Set.univ) (tsum fun i => μ (HSMul.hSMul i s))
      -/
      exact h_measure_univ_le }
      /-
        🎉 no goals
      -/


@[to_additive]
theorem iUnion_smul_ae_eq (h : IsFundamentalDomain G s μ) : ⋃ g : G, g • s =ᵐ[μ] univ :=
  eventuallyEq_univ.2 <| h.ae_covers.mono fun _ ⟨g, hg⟩ =>
    mem_iUnion.2 ⟨g⁻¹, _, hg, inv_smul_smul _ _⟩


@[to_additive]
theorem measure_ne_zero [Countable G] [SMulInvariantMeasure G α μ]
    (hμ : μ ≠ 0) (h : IsFundamentalDomain G s μ) : μ s ≠ 0 := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁴ : Group G
    inst✝³ : MulAction G α
    inst✝² : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝¹ : Countable G
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    hμ : Ne μ 0
    h : MeasureTheory.IsFundamentalDomain G s μ
    ⊢ Ne (μ s) 0
  -/
  have hc := measure_univ_pos.mpr hμ
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁴ : Group G
    inst✝³ : MulAction G α
    inst✝² : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝¹ : Countable G
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    hμ : Ne μ 0
    h : MeasureTheory.IsFundamentalDomain G s μ
    hc : LT.lt 0 (μ Set.univ)
    ⊢ Ne (μ s) 0
  -/
  contrapose! hc
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁴ : Group G
    inst✝³ : MulAction G α
    inst✝² : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝¹ : Countable G
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    hμ : Ne μ 0
    h : MeasureTheory.IsFundamentalDomain G s μ
    hc : Eq (μ s) 0
    ⊢ LE.le (μ Set.univ) 0
  -/
  rw [← measure_congr h.iUnion_smul_ae_eq]
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁴ : Group G
    inst✝³ : MulAction G α
    inst✝² : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝¹ : Countable G
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    hμ : Ne μ 0
    h : MeasureTheory.IsFundamentalDomain G s μ
    hc : Eq (μ s) 0
    ⊢ LE.le (μ (Set.iUnion fun g => HSMul.hSMul g s)) 0
  -/
  refine le_trans (measure_iUnion_le _) ?_
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁴ : Group G
    inst✝³ : MulAction G α
    inst✝² : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝¹ : Countable G
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    hμ : Ne μ 0
    h : MeasureTheory.IsFundamentalDomain G s μ
    hc : Eq (μ s) 0
    ⊢ LE.le (tsum fun i => μ (HSMul.hSMul i s)) 0
  -/
  simp_rw [measure_smul, hc, tsum_zero, le_refl]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mono (h : IsFundamentalDomain G s μ) {ν : Measure α} (hle : ν ≪ μ) :
    IsFundamentalDomain G s ν :=
  ⟨h.1.mono_ac hle, hle h.2, h.aedisjoint.mono fun _ _ h => hle h⟩


@[to_additive]
theorem preimage_of_equiv {ν : Measure β} (h : IsFundamentalDomain G s μ) {f : β → α}
    (hf : QuasiMeasurePreserving f ν μ) {e : G → H} (he : Bijective e)
    (hef : ∀ g, Semiconj f (e g • ·) (g • ·)) : IsFundamentalDomain H (f ⁻¹' s) ν where
  nullMeasurableSet := h.nullMeasurableSet.preimage hf
                                                                  /-
                                                                    G : Type u_1
                                                                    H : Type u_2
                                                                    α : Type u_3
                                                                    β : Type u_4
                                                                    inst✝⁵ : Group G
                                                                    inst✝⁴ : Group H
                                                                    inst✝³ : MulAction G α
                                                                    inst✝² : MeasurableSpace α
                                                                    inst✝¹ : MulAction H β
                                                                    inst✝ : MeasurableSpace β
                                                                    s : Set α
                                                                    μ : MeasureTheory.Measure α
                                                                    ν : MeasureTheory.Measure β
                                                                    h : MeasureTheory.IsFundamentalDomain G s μ
                                                                    f : β → α
                                                                    hf : MeasureTheory.Measure.QuasiMeasurePreserving f ν μ
                                                                    e : G → H
                                                                    he : Function.Bijective e
                                                                    hef : ∀ (g : G), Function.Semiconj f (fun x => HSMul.hSMul (e g) x) fun x => H …
                                                                    x : β
                                                                    x✝ : Exists fun g => Membership.mem s (HSMul.hSMul g (f x))
                                                                    g : G
                                                                    hg : Membership.mem s (HSMul.hSMul g (f x))
                                                                    ⊢ Membership.mem (Set.preimage f s) (HSMul.hSMul (e g) x)
                                                                  -/
  ae_covers := (hf.ae h.ae_covers).mono fun x ⟨g, hg⟩ => ⟨e g, by rwa [mem_preimage, hef g x]⟩
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
  aedisjoint a b hab := by
    /-
      G : Type u_1
      H : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝⁵ : Group G
      inst✝⁴ : Group H
      inst✝³ : MulAction G α
      inst✝² : MeasurableSpace α
      inst✝¹ : MulAction H β
      inst✝ : MeasurableSpace β
      s : Set α
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      h : MeasureTheory.IsFundamentalDomain G s μ
      f : β → α
      hf : MeasureTheory.Measure.QuasiMeasurePreserving f ν μ
      e : G → H
      he : Function.Bijective e
      hef : ∀ (g : G), Function.Semiconj f (fun x => HSMul.hSMul (e g) x) fun x => H …
      a b : H
      hab : Ne a b
      ⊢ Function.onFun (MeasureTheory.AEDisjoint ν) (fun g => HSMul.hSMul g (Set.pre …
    -/
    lift e to G ≃ H using he
    /-
      case intro
      G : Type u_1
      H : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝⁵ : Group G
      inst✝⁴ : Group H
      inst✝³ : MulAction G α
      inst✝² : MeasurableSpace α
      inst✝¹ : MulAction H β
      inst✝ : MeasurableSpace β
      s : Set α
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      h : MeasureTheory.IsFundamentalDomain G s μ
      f : β → α
      hf : MeasureTheory.Measure.QuasiMeasurePreserving f ν μ
      a b : H
      hab : Ne a b
      e : Equiv G H
      hef : ∀ (g : G), Function.Semiconj f (fun x => HSMul.hSMul (e g) x) fun x => H …
      ⊢ Function.onFun (MeasureTheory.AEDisjoint ν) (fun g => HSMul.hSMul g (Set.pre …
    -/
    have : (e.symm a⁻¹)⁻¹ ≠ (e.symm b⁻¹)⁻¹ := by simp [hab]
    /-
      case intro
      G : Type u_1
      H : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝⁵ : Group G
      inst✝⁴ : Group H
      inst✝³ : MulAction G α
      inst✝² : MeasurableSpace α
      inst✝¹ : MulAction H β
      inst✝ : MeasurableSpace β
      s : Set α
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      h : MeasureTheory.IsFundamentalDomain G s μ
      f : β → α
      hf : MeasureTheory.Measure.QuasiMeasurePreserving f ν μ
      a b : H
      hab : Ne a b
      e : Equiv G H
      hef : ∀ (g : G), Function.Semiconj f (fun x => HSMul.hSMul (e g) x) fun x => H …
      this : Ne (Inv.inv (e.symm (Inv.inv a))) (Inv.inv (e.symm (Inv.inv b)))
      ⊢ Function.onFun (MeasureTheory.AEDisjoint ν) (fun g => HSMul.hSMul g (Set.pre …
    -/
    have := (h.aedisjoint this).preimage hf
    /-
      case intro
      G : Type u_1
      H : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝⁵ : Group G
      inst✝⁴ : Group H
      inst✝³ : MulAction G α
      inst✝² : MeasurableSpace α
      inst✝¹ : MulAction H β
      inst✝ : MeasurableSpace β
      s : Set α
      μ : MeasureTheory.Measure α
      ν : MeasureTheory.Measure β
      h : MeasureTheory.IsFundamentalDomain G s μ
      f : β → α
      hf : MeasureTheory.Measure.QuasiMeasurePreserving f ν μ
      a b : H
      hab : Ne a b
      e : Equiv G H
      hef : ∀ (g : G), Function.Semiconj f (fun x => HSMul.hSMul (e g) x) fun x => H …
      this✝ : Ne (Inv.inv (e.symm (Inv.inv a))) (Inv.inv (e.symm (Inv.inv b)))
      this : MeasureTheory.AEDisjoint ν (Set.preimage f ((fun g => HSMul.hSMul g s)  …
      ⊢ Function.onFun (MeasureTheory.AEDisjoint ν) (fun g => HSMul.hSMul g (Set.pre …
    -/
    simp only [Semiconj] at hef
    simpa only [onFun, ← preimage_smul_inv, preimage_preimage, ← hef, e.apply_symm_apply, inv_inv]
      using this


@[to_additive]
theorem image_of_equiv {ν : Measure β} (h : IsFundamentalDomain G s μ) (f : α ≃ β)
    (hf : QuasiMeasurePreserving f.symm ν μ) (e : H ≃ G)
    (hef : ∀ g, Semiconj f (e g • ·) (g • ·)) : IsFundamentalDomain H (f '' s) ν := by
  /-
    G : Type u_1
    H : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝⁵ : Group G
    inst✝⁴ : Group H
    inst✝³ : MulAction G α
    inst✝² : MeasurableSpace α
    inst✝¹ : MulAction H β
    inst✝ : MeasurableSpace β
    s : Set α
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    h : MeasureTheory.IsFundamentalDomain G s μ
    f : Equiv α β
    hf : MeasureTheory.Measure.QuasiMeasurePreserving (⇑f.symm) ν μ
    e : Equiv H G
    hef : ∀ (g : H), Function.Semiconj (⇑f) (fun x => HSMul.hSMul (e g) x) fun x = …
    ⊢ MeasureTheory.IsFundamentalDomain H (Set.image (⇑f) s) ν
  -/
  rw [f.image_eq_preimage]
  /-
    G : Type u_1
    H : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝⁵ : Group G
    inst✝⁴ : Group H
    inst✝³ : MulAction G α
    inst✝² : MeasurableSpace α
    inst✝¹ : MulAction H β
    inst✝ : MeasurableSpace β
    s : Set α
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    h : MeasureTheory.IsFundamentalDomain G s μ
    f : Equiv α β
    hf : MeasureTheory.Measure.QuasiMeasurePreserving (⇑f.symm) ν μ
    e : Equiv H G
    hef : ∀ (g : H), Function.Semiconj (⇑f) (fun x => HSMul.hSMul (e g) x) fun x = …
    ⊢ MeasureTheory.IsFundamentalDomain H (Set.preimage (⇑f.symm) s) ν
  -/
  refine h.preimage_of_equiv hf e.symm.bijective fun g x => ?_
  /-
    G : Type u_1
    H : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝⁵ : Group G
    inst✝⁴ : Group H
    inst✝³ : MulAction G α
    inst✝² : MeasurableSpace α
    inst✝¹ : MulAction H β
    inst✝ : MeasurableSpace β
    s : Set α
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    h : MeasureTheory.IsFundamentalDomain G s μ
    f : Equiv α β
    hf : MeasureTheory.Measure.QuasiMeasurePreserving (⇑f.symm) ν μ
    e : Equiv H G
    hef : ∀ (g : H), Function.Semiconj (⇑f) (fun x => HSMul.hSMul (e g) x) fun x = …
    g : G
    x : β
    ⊢ Eq (f.symm ((fun x => HSMul.hSMul (e.symm g) x) x)) ((fun x => HSMul.hSMul g …
  -/
  rcases f.surjective x with ⟨x, rfl⟩
  /-
    case intro
    G : Type u_1
    H : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝⁵ : Group G
    inst✝⁴ : Group H
    inst✝³ : MulAction G α
    inst✝² : MeasurableSpace α
    inst✝¹ : MulAction H β
    inst✝ : MeasurableSpace β
    s : Set α
    μ : MeasureTheory.Measure α
    ν : MeasureTheory.Measure β
    h : MeasureTheory.IsFundamentalDomain G s μ
    f : Equiv α β
    hf : MeasureTheory.Measure.QuasiMeasurePreserving (⇑f.symm) ν μ
    e : Equiv H G
    hef : ∀ (g : H), Function.Semiconj (⇑f) (fun x => HSMul.hSMul (e g) x) fun x = …
    g : G
    x : α
    ⊢ Eq (f.symm ((fun x => HSMul.hSMul (e.symm g) x) (f x))) ((fun x => HSMul.hSM …
  -/
  rw [← hef _ _, f.symm_apply_apply, f.symm_apply_apply, e.apply_symm_apply]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem pairwise_aedisjoint_of_ac {ν} (h : IsFundamentalDomain G s μ) (hν : ν ≪ μ) :
    Pairwise fun g₁ g₂ : G => AEDisjoint ν (g₁ • s) (g₂ • s) :=
  h.aedisjoint.mono fun _ _ H => hν H


@[to_additive]
theorem smul_of_comm {G' : Type*} [Group G'] [MulAction G' α] [MeasurableSpace G']
    [MeasurableSMul G' α] [SMulInvariantMeasure G' α μ] [SMulCommClass G' G α]
    (h : IsFundamentalDomain G s μ) (g : G') : IsFundamentalDomain G (g • s) μ :=
  h.image_of_equiv (MulAction.toPerm g) (measurePreserving_smul _ _).quasiMeasurePreserving
    (Equiv.refl _) <| smul_comm g


@[to_additive]
theorem nullMeasurableSet_smul (h : IsFundamentalDomain G s μ) (g : G) :
    NullMeasurableSet (g • s) μ :=
  h.nullMeasurableSet.smul g


@[to_additive]
theorem restrict_restrict (h : IsFundamentalDomain G s μ) (g : G) (t : Set α) :
    (μ.restrict t).restrict (g • s) = μ.restrict (g • s ∩ t) :=
  restrict_restrict₀ ((h.nullMeasurableSet_smul g).mono restrict_le_self)


@[to_additive]
theorem smul (h : IsFundamentalDomain G s μ) (g : G) : IsFundamentalDomain G (g • s) μ :=
  h.image_of_equiv (MulAction.toPerm g) (measurePreserving_smul _ _).quasiMeasurePreserving
                                                                  /-
                                                                    G : Type u_1
                                                                    α : Type u_3
                                                                    inst✝⁵ : Group G
                                                                    inst✝⁴ : MulAction G α
                                                                    inst✝³ : MeasurableSpace α
                                                                    s : Set α
                                                                    μ : MeasureTheory.Measure α
                                                                    inst✝² : MeasurableSpace G
                                                                    inst✝¹ : MeasurableSMul G α
                                                                    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
                                                                    h : MeasureTheory.IsFundamentalDomain G s μ
                                                                    g g' : G
                                                                    ⊢ Eq ((fun g' => HMul.hMul (HMul.hMul g g') (Inv.inv g)) ((fun g' => HMul.hMul …
                                                                  -/
    ⟨fun g' => g⁻¹ * g' * g, fun g' => g * g' * g⁻¹, fun g' => by simp [mul_assoc], fun g' => by
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
      /-
        G : Type u_1
        α : Type u_3
        inst✝⁵ : Group G
        inst✝⁴ : MulAction G α
        inst✝³ : MeasurableSpace α
        s : Set α
        μ : MeasureTheory.Measure α
        inst✝² : MeasurableSpace G
        inst✝¹ : MeasurableSMul G α
        inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
        h : MeasureTheory.IsFundamentalDomain G s μ
        g g' : G
        ⊢ Eq ((fun g' => HMul.hMul (HMul.hMul (Inv.inv g) g') g) ((fun g' => HMul.hMul …
      -/
      simp [mul_assoc]⟩
      /-
        🎉 no goals
      -/
                   /-
                     G : Type u_1
                     α : Type u_3
                     inst✝⁵ : Group G
                     inst✝⁴ : MulAction G α
                     inst✝³ : MeasurableSpace α
                     s : Set α
                     μ : MeasureTheory.Measure α
                     inst✝² : MeasurableSpace G
                     inst✝¹ : MeasurableSMul G α
                     inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
                     h : MeasureTheory.IsFundamentalDomain G s μ
                     g g' : G
                     x : α
                     ⊢ Eq ((MulAction.toPerm g) ((fun x => HSMul.hSMul ({ toFun := fun g' => HMul.h …
                   -/
    fun g' x => by simp [smul_smul, mul_assoc]
                   /-
                     🎉 no goals
                   -/


@[to_additive]
theorem sum_restrict_of_ac (h : IsFundamentalDomain G s μ) (hν : ν ≪ μ) :
    (sum fun g : G => ν.restrict (g • s)) = ν := by
  rw [← restrict_iUnion_ae (h.aedisjoint.mono fun i j h => hν h) fun g =>
      (h.nullMeasurableSet_smul g).mono_ac hν,
    restrict_congr_set (hν h.iUnion_smul_ae_eq), restrict_univ]


@[to_additive]
theorem lintegral_eq_tsum_of_ac (h : IsFundamentalDomain G s μ) (hν : ν ≪ μ) (f : α → ℝ≥0∞) :
    ∫⁻ x, f x ∂ν = ∑' g : G, ∫⁻ x in g • s, f x ∂ν := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    ν : MeasureTheory.Measure α
    h : MeasureTheory.IsFundamentalDomain G s μ
    hν : ν.AbsolutelyContinuous μ
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral ν fun x => f x) (tsum fun g => MeasureTheory.lin …
  -/
  rw [← lintegral_sum_measure, h.sum_restrict_of_ac hν]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem sum_restrict (h : IsFundamentalDomain G s μ) : (sum fun g : G => μ.restrict (g • s)) = μ :=
  h.sum_restrict_of_ac (refl _)


@[to_additive]
theorem lintegral_eq_tsum (h : IsFundamentalDomain G s μ) (f : α → ℝ≥0∞) :
    ∫⁻ x, f x ∂μ = ∑' g : G, ∫⁻ x in g • s, f x ∂μ :=
  h.lintegral_eq_tsum_of_ac (refl _) f


@[to_additive]
theorem lintegral_eq_tsum' (h : IsFundamentalDomain G s μ) (f : α → ℝ≥0∞) :
    ∫⁻ x, f x ∂μ = ∑' g : G, ∫⁻ x in s, f (g⁻¹ • x) ∂μ :=
  calc
    ∫⁻ x, f x ∂μ = ∑' g : G, ∫⁻ x in g • s, f x ∂μ := h.lintegral_eq_tsum f
    _ = ∑' g : G, ∫⁻ x in g⁻¹ • s, f x ∂μ := ((Equiv.inv G).tsum_eq _).symm
    _ = ∑' g : G, ∫⁻ x in s, f (g⁻¹ • x) ∂μ := tsum_congr fun g => Eq.symm <|
      (measurePreserving_smul g⁻¹ μ).setLIntegral_comp_emb (measurableEmbedding_const_smul _) _ _


@[to_additive] lemma lintegral_eq_tsum'' (h : IsFundamentalDomain G s μ) (f : α → ℝ≥0∞) :
    ∫⁻ x, f x ∂μ = ∑' g : G, ∫⁻ x in s, f (g • x) ∂μ :=
  (lintegral_eq_tsum' h f).trans ((Equiv.inv G).tsum_eq (fun g ↦ ∫⁻ (x : α) in s, f (g • x) ∂μ))


@[to_additive]
theorem setLIntegral_eq_tsum (h : IsFundamentalDomain G s μ) (f : α → ℝ≥0∞) (t : Set α) :
    ∫⁻ x in t, f x ∂μ = ∑' g : G, ∫⁻ x in t ∩ g • s, f x ∂μ :=
  calc
    ∫⁻ x in t, f x ∂μ = ∑' g : G, ∫⁻ x in g • s, f x ∂μ.restrict t :=
      h.lintegral_eq_tsum_of_ac restrict_le_self.absolutelyContinuous _
                                                  /-
                                                    G : Type u_1
                                                    α : Type u_3
                                                    inst✝⁶ : Group G
                                                    inst✝⁵ : MulAction G α
                                                    inst✝⁴ : MeasurableSpace α
                                                    s : Set α
                                                    μ : MeasureTheory.Measure α
                                                    inst✝³ : MeasurableSpace G
                                                    inst✝² : MeasurableSMul G α
                                                    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
                                                    inst✝ : Countable G
                                                    h : MeasureTheory.IsFundamentalDomain G s μ
                                                    f : α → ENNReal
                                                    t : Set α
                                                    ⊢ Eq (tsum fun g => MeasureTheory.lintegral ((μ.restrict t).restrict (HSMul.hS …
                                                  -/
    _ = ∑' g : G, ∫⁻ x in t ∩ g • s, f x ∂μ := by simp only [h.restrict_restrict, inter_comm]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_eq_tsum := setLIntegral_eq_tsum


@[to_additive]
theorem setLIntegral_eq_tsum' (h : IsFundamentalDomain G s μ) (f : α → ℝ≥0∞) (t : Set α) :
    ∫⁻ x in t, f x ∂μ = ∑' g : G, ∫⁻ x in g • t ∩ s, f (g⁻¹ • x) ∂μ :=
  calc
    ∫⁻ x in t, f x ∂μ = ∑' g : G, ∫⁻ x in t ∩ g • s, f x ∂μ := h.setLIntegral_eq_tsum f t
    _ = ∑' g : G, ∫⁻ x in t ∩ g⁻¹ • s, f x ∂μ := ((Equiv.inv G).tsum_eq _).symm
                                                          /-
                                                            G : Type u_1
                                                            α : Type u_3
                                                            inst✝⁶ : Group G
                                                            inst✝⁵ : MulAction G α
                                                            inst✝⁴ : MeasurableSpace α
                                                            s : Set α
                                                            μ : MeasureTheory.Measure α
                                                            inst✝³ : MeasurableSpace G
                                                            inst✝² : MeasurableSMul G α
                                                            inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
                                                            inst✝ : Countable G
                                                            h : MeasureTheory.IsFundamentalDomain G s μ
                                                            f : α → ENNReal
                                                            t : Set α
                                                            ⊢ Eq (tsum fun g => MeasureTheory.lintegral (μ.restrict (Inter.inter t (HSMul. …
                                                          -/
    _ = ∑' g : G, ∫⁻ x in g⁻¹ • (g • t ∩ s), f x ∂μ := by simp only [smul_set_inter, inv_smul_smul]
                                                          /-
                                                            🎉 no goals
                                                          -/
    _ = ∑' g : G, ∫⁻ x in g • t ∩ s, f (g⁻¹ • x) ∂μ := tsum_congr fun g => Eq.symm <|
      (measurePreserving_smul g⁻¹ μ).setLIntegral_comp_emb (measurableEmbedding_const_smul _) _ _


@[deprecated (since := "2024-06-29")]
alias set_lintegral_eq_tsum' := setLIntegral_eq_tsum'


@[to_additive]
theorem measure_eq_tsum_of_ac (h : IsFundamentalDomain G s μ) (hν : ν ≪ μ) (t : Set α) :
    ν t = ∑' g : G, ν (t ∩ g • s) := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    ν : MeasureTheory.Measure α
    h : MeasureTheory.IsFundamentalDomain G s μ
    hν : ν.AbsolutelyContinuous μ
    t : Set α
    ⊢ Eq (ν t) (tsum fun g => ν (Inter.inter t (HSMul.hSMul g s)))
  -/
  have H : ν.restrict t ≪ μ := Measure.restrict_le_self.absolutelyContinuous.trans hν
  simpa only [setLIntegral_one, Pi.one_def,
    Measure.restrict_apply₀ ((h.nullMeasurableSet_smul _).mono_ac H), inter_comm] using
    h.lintegral_eq_tsum_of_ac H 1


@[to_additive]
theorem measure_eq_tsum' (h : IsFundamentalDomain G s μ) (t : Set α) :
    μ t = ∑' g : G, μ (t ∩ g • s) :=
  h.measure_eq_tsum_of_ac AbsolutelyContinuous.rfl t


@[to_additive]
theorem measure_eq_tsum (h : IsFundamentalDomain G s μ) (t : Set α) :
    μ t = ∑' g : G, μ (g • t ∩ s) := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    h : MeasureTheory.IsFundamentalDomain G s μ
    t : Set α
    ⊢ Eq (μ t) (tsum fun g => μ (Inter.inter (HSMul.hSMul g t) s))
  -/
  simpa only [setLIntegral_one] using h.setLIntegral_eq_tsum' (fun _ => 1) t
  /-
    🎉 no goals
  -/


@[to_additive]
theorem measure_zero_of_invariant (h : IsFundamentalDomain G s μ) (t : Set α)
    (ht : ∀ g : G, g • t = t) (hts : μ (t ∩ s) = 0) : μ t = 0 := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    h : MeasureTheory.IsFundamentalDomain G s μ
    t : Set α
    ht : ∀ (g : G), Eq (HSMul.hSMul g t) t
    hts : Eq (μ (Inter.inter t s)) 0
    ⊢ Eq (μ t) 0
  -/
  rw [measure_eq_tsum h]; simp [ht, hts]
                          /-
                            🎉 no goals
                          -/


/-- Given a measure space with an action of a finite group `G`, the measure of any `G`-invariant set
is determined by the measure of its intersection with a fundamental domain for the action of `G`. -/
@[to_additive measure_eq_card_smul_of_vadd_ae_eq_self "Given a measure space with an action of a
  finite additive group `G`, the measure of any `G`-invariant set is determined by the measure of
  its intersection with a fundamental domain for the action of `G`."]
theorem measure_eq_card_smul_of_smul_ae_eq_self [Finite G] (h : IsFundamentalDomain G s μ)
    (t : Set α) (ht : ∀ g : G, (g • t : Set α) =ᵐ[μ] t) : μ t = Nat.card G • μ (t ∩ s) := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁷ : Group G
    inst✝⁶ : MulAction G α
    inst✝⁵ : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝⁴ : MeasurableSpace G
    inst✝³ : MeasurableSMul G α
    inst✝² : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝¹ : Countable G
    inst✝ : Finite G
    h : MeasureTheory.IsFundamentalDomain G s μ
    t : Set α
    ht : ∀ (g : G), (MeasureTheory.ae μ).EventuallyEq (HSMul.hSMul g t) t
    ⊢ Eq (μ t) (HSMul.hSMul (Nat.card G) (μ (Inter.inter t s)))
  -/
  haveI : Fintype G := Fintype.ofFinite G
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁷ : Group G
    inst✝⁶ : MulAction G α
    inst✝⁵ : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝⁴ : MeasurableSpace G
    inst✝³ : MeasurableSMul G α
    inst✝² : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝¹ : Countable G
    inst✝ : Finite G
    h : MeasureTheory.IsFundamentalDomain G s μ
    t : Set α
    ht : ∀ (g : G), (MeasureTheory.ae μ).EventuallyEq (HSMul.hSMul g t) t
    this : Fintype G
    ⊢ Eq (μ t) (HSMul.hSMul (Nat.card G) (μ (Inter.inter t s)))
  -/
  rw [h.measure_eq_tsum]
  replace ht : ∀ g : G, (g • t ∩ s : Set α) =ᵐ[μ] (t ∩ s : Set α) := fun g =>
    ae_eq_set_inter (ht g) (ae_eq_refl s)
  simp_rw [measure_congr (ht _), tsum_fintype, Finset.sum_const, Nat.card_eq_fintype_card,
    Finset.card_univ]


@[to_additive]
protected theorem setLIntegral_eq (hs : IsFundamentalDomain G s μ) (ht : IsFundamentalDomain G t μ)
    (f : α → ℝ≥0∞) (hf : ∀ (g : G) (x), f (g • x) = f x) :
    ∫⁻ x in s, f x ∂μ = ∫⁻ x in t, f x ∂μ :=
  calc
    ∫⁻ x in s, f x ∂μ = ∑' g : G, ∫⁻ x in s ∩ g • t, f x ∂μ := ht.setLIntegral_eq_tsum _ _
                                                          /-
                                                            G : Type u_1
                                                            α : Type u_3
                                                            inst✝⁶ : Group G
                                                            inst✝⁵ : MulAction G α
                                                            inst✝⁴ : MeasurableSpace α
                                                            s t : Set α
                                                            μ : MeasureTheory.Measure α
                                                            inst✝³ : MeasurableSpace G
                                                            inst✝² : MeasurableSMul G α
                                                            inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
                                                            inst✝ : Countable G
                                                            hs : MeasureTheory.IsFundamentalDomain G s μ
                                                            ht : MeasureTheory.IsFundamentalDomain G t μ
                                                            f : α → ENNReal
                                                            hf : ∀ (g : G) (x : α), Eq (f (HSMul.hSMul g x)) (f x)
                                                            ⊢ Eq (tsum fun g => MeasureTheory.lintegral (μ.restrict (Inter.inter s (HSMul. …
                                                          -/
    _ = ∑' g : G, ∫⁻ x in g • t ∩ s, f (g⁻¹ • x) ∂μ := by simp only [hf, inter_comm]
                                                          /-
                                                            🎉 no goals
                                                          -/
    _ = ∫⁻ x in t, f x ∂μ := (hs.setLIntegral_eq_tsum' _ _).symm


@[deprecated (since := "2024-06-29")]
alias set_lintegral_eq := MeasureTheory.IsFundamentalDomain.setLIntegral_eq


@[to_additive]
theorem measure_set_eq (hs : IsFundamentalDomain G s μ) (ht : IsFundamentalDomain G t μ) {A : Set α}
    (hA₀ : MeasurableSet A) (hA : ∀ g : G, (fun x => g • x) ⁻¹' A = A) : μ (A ∩ s) = μ (A ∩ t) := by
  have : ∫⁻ x in s, A.indicator 1 x ∂μ = ∫⁻ x in t, A.indicator 1 x ∂μ := by
    refine hs.setLIntegral_eq ht (Set.indicator A fun _ => 1) fun g x ↦ ?_
    convert (Set.indicator_comp_right (g • · : α → α) (g := fun _ ↦ (1 : ℝ≥0∞))).symm
    rw [hA g]
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s t : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    ht : MeasureTheory.IsFundamentalDomain G t μ
    A : Set α
    hA₀ : MeasurableSet A
    hA : ∀ (g : G), Eq (Set.preimage (fun x => HSMul.hSMul g x) A) A
    this : Eq (MeasureTheory.lintegral (μ.restrict s) fun x => A.indicator 1 x) (M …
    ⊢ Eq (μ (Inter.inter A s)) (μ (Inter.inter A t))
  -/
  simpa [Measure.restrict_apply hA₀, lintegral_indicator hA₀] using this
  /-
    🎉 no goals
  -/


/-- If `s` and `t` are two fundamental domains of the same action, then their measures are equal. -/
@[to_additive "If `s` and `t` are two fundamental domains of the same action, then their measures
  are equal."]
protected theorem measure_eq (hs : IsFundamentalDomain G s μ) (ht : IsFundamentalDomain G t μ) :
    μ s = μ t := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s t : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    ht : MeasureTheory.IsFundamentalDomain G t μ
    ⊢ Eq (μ s) (μ t)
  -/
  simpa only [setLIntegral_one] using hs.setLIntegral_eq ht (fun _ => 1) fun _ _ => rfl
  /-
    🎉 no goals
  -/


@[to_additive]
protected theorem aEStronglyMeasurable_on_iff {β : Type*} [TopologicalSpace β]
    [PseudoMetrizableSpace β] (hs : IsFundamentalDomain G s μ) (ht : IsFundamentalDomain G t μ)
    {f : α → β} (hf : ∀ (g : G) (x), f (g • x) = f x) :
    AEStronglyMeasurable f (μ.restrict s) ↔ AEStronglyMeasurable f (μ.restrict t) :=
  calc
    AEStronglyMeasurable f (μ.restrict s) ↔
        AEStronglyMeasurable f (Measure.sum fun g : G => μ.restrict (g • t ∩ s)) := by
      simp only [← ht.restrict_restrict,
        ht.sum_restrict_of_ac restrict_le_self.absolutelyContinuous]
    _ ↔ ∀ g : G, AEStronglyMeasurable f (μ.restrict (g • (g⁻¹ • s ∩ t))) := by
      /-
        G : Type u_1
        α : Type u_3
        inst✝⁸ : Group G
        inst✝⁷ : MulAction G α
        inst✝⁶ : MeasurableSpace α
        s t : Set α
        μ : MeasureTheory.Measure α
        inst✝⁵ : MeasurableSpace G
        inst✝⁴ : MeasurableSMul G α
        inst✝³ : MeasureTheory.SMulInvariantMeasure G α μ
        inst✝² : Countable G
        β : Type u_6
        inst✝¹ : TopologicalSpace β
        inst✝ : TopologicalSpace.PseudoMetrizableSpace β
        hs : MeasureTheory.IsFundamentalDomain G s μ
        ht : MeasureTheory.IsFundamentalDomain G t μ
        f : α → β
        hf : ∀ (g : G) (x : α), Eq (f (HSMul.hSMul g x)) (f x)
        ⊢ Iff (MeasureTheory.AEStronglyMeasurable f (MeasureTheory.Measure.sum fun g = …
      -/
      simp only [smul_set_inter, inter_comm, smul_inv_smul, aestronglyMeasurable_sum_measure_iff]
      /-
        🎉 no goals
      -/
    _ ↔ ∀ g : G, AEStronglyMeasurable f (μ.restrict (g⁻¹ • (g⁻¹⁻¹ • s ∩ t))) :=
      inv_surjective.forall
                                                                               /-
                                                                                 G : Type u_1
                                                                                 α : Type u_3
                                                                                 inst✝⁸ : Group G
                                                                                 inst✝⁷ : MulAction G α
                                                                                 inst✝⁶ : MeasurableSpace α
                                                                                 s t : Set α
                                                                                 μ : MeasureTheory.Measure α
                                                                                 inst✝⁵ : MeasurableSpace G
                                                                                 inst✝⁴ : MeasurableSMul G α
                                                                                 inst✝³ : MeasureTheory.SMulInvariantMeasure G α μ
                                                                                 inst✝² : Countable G
                                                                                 β : Type u_6
                                                                                 inst✝¹ : TopologicalSpace β
                                                                                 inst✝ : TopologicalSpace.PseudoMetrizableSpace β
                                                                                 hs : MeasureTheory.IsFundamentalDomain G s μ
                                                                                 ht : MeasureTheory.IsFundamentalDomain G t μ
                                                                                 f : α → β
                                                                                 hf : ∀ (g : G) (x : α), Eq (f (HSMul.hSMul g x)) (f x)
                                                                                 ⊢ Iff (∀ (g : G), MeasureTheory.AEStronglyMeasurable f (μ.restrict (HSMul.hSMu …
                                                                               -/
    _ ↔ ∀ g : G, AEStronglyMeasurable f (μ.restrict (g⁻¹ • (g • s ∩ t))) := by simp only [inv_inv]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
    _ ↔ ∀ g : G, AEStronglyMeasurable f (μ.restrict (g • s ∩ t)) := by
      /-
        G : Type u_1
        α : Type u_3
        inst✝⁸ : Group G
        inst✝⁷ : MulAction G α
        inst✝⁶ : MeasurableSpace α
        s t : Set α
        μ : MeasureTheory.Measure α
        inst✝⁵ : MeasurableSpace G
        inst✝⁴ : MeasurableSMul G α
        inst✝³ : MeasureTheory.SMulInvariantMeasure G α μ
        inst✝² : Countable G
        β : Type u_6
        inst✝¹ : TopologicalSpace β
        inst✝ : TopologicalSpace.PseudoMetrizableSpace β
        hs : MeasureTheory.IsFundamentalDomain G s μ
        ht : MeasureTheory.IsFundamentalDomain G t μ
        f : α → β
        hf : ∀ (g : G) (x : α), Eq (f (HSMul.hSMul g x)) (f x)
        ⊢ Iff (∀ (g : G), MeasureTheory.AEStronglyMeasurable f (μ.restrict (HSMul.hSMu …
      -/
      refine forall_congr' fun g => ?_
      /-
        G : Type u_1
        α : Type u_3
        inst✝⁸ : Group G
        inst✝⁷ : MulAction G α
        inst✝⁶ : MeasurableSpace α
        s t : Set α
        μ : MeasureTheory.Measure α
        inst✝⁵ : MeasurableSpace G
        inst✝⁴ : MeasurableSMul G α
        inst✝³ : MeasureTheory.SMulInvariantMeasure G α μ
        inst✝² : Countable G
        β : Type u_6
        inst✝¹ : TopologicalSpace β
        inst✝ : TopologicalSpace.PseudoMetrizableSpace β
        hs : MeasureTheory.IsFundamentalDomain G s μ
        ht : MeasureTheory.IsFundamentalDomain G t μ
        f : α → β
        hf : ∀ (g : G) (x : α), Eq (f (HSMul.hSMul g x)) (f x)
        g : G
        ⊢ Iff (MeasureTheory.AEStronglyMeasurable f (μ.restrict (HSMul.hSMul (Inv.inv  …
      -/
      have he : MeasurableEmbedding (g⁻¹ • · : α → α) := measurableEmbedding_const_smul _
      rw [← image_smul, ← ((measurePreserving_smul g⁻¹ μ).restrict_image_emb he
        _).aestronglyMeasurable_comp_iff he]
      /-
        G : Type u_1
        α : Type u_3
        inst✝⁸ : Group G
        inst✝⁷ : MulAction G α
        inst✝⁶ : MeasurableSpace α
        s t : Set α
        μ : MeasureTheory.Measure α
        inst✝⁵ : MeasurableSpace G
        inst✝⁴ : MeasurableSMul G α
        inst✝³ : MeasureTheory.SMulInvariantMeasure G α μ
        inst✝² : Countable G
        β : Type u_6
        inst✝¹ : TopologicalSpace β
        inst✝ : TopologicalSpace.PseudoMetrizableSpace β
        hs : MeasureTheory.IsFundamentalDomain G s μ
        ht : MeasureTheory.IsFundamentalDomain G t μ
        f : α → β
        hf : ∀ (g : G) (x : α), Eq (f (HSMul.hSMul g x)) (f x)
        g : G
        he : MeasurableEmbedding fun x => HSMul.hSMul (Inv.inv g) x
        ⊢ Iff (MeasureTheory.AEStronglyMeasurable (Function.comp f fun x => HSMul.hSMu …
      -/
      simp only [Function.comp_def, hf]
      /-
        🎉 no goals
      -/
    _ ↔ AEStronglyMeasurable f (μ.restrict t) := by
      simp only [← aestronglyMeasurable_sum_measure_iff, ← hs.restrict_restrict,
        hs.sum_restrict_of_ac restrict_le_self.absolutelyContinuous]


@[to_additive]
protected theorem hasFiniteIntegral_on_iff (hs : IsFundamentalDomain G s μ)
    (ht : IsFundamentalDomain G t μ) {f : α → E} (hf : ∀ (g : G) (x), f (g • x) = f x) :
    HasFiniteIntegral f (μ.restrict s) ↔ HasFiniteIntegral f (μ.restrict t) := by
  /-
    G : Type u_1
    α : Type u_3
    E : Type u_5
    inst✝⁷ : Group G
    inst✝⁶ : MulAction G α
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : NormedAddCommGroup E
    s t : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    ht : MeasureTheory.IsFundamentalDomain G t μ
    f : α → E
    hf : ∀ (g : G) (x : α), Eq (f (HSMul.hSMul g x)) (f x)
    ⊢ Iff (MeasureTheory.HasFiniteIntegral f (μ.restrict s)) (MeasureTheory.HasFin …
  -/
  dsimp only [HasFiniteIntegral]
  /-
    G : Type u_1
    α : Type u_3
    E : Type u_5
    inst✝⁷ : Group G
    inst✝⁶ : MulAction G α
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : NormedAddCommGroup E
    s t : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    ht : MeasureTheory.IsFundamentalDomain G t μ
    f : α → E
    hf : ∀ (g : G) (x : α), Eq (f (HSMul.hSMul g x)) (f x)
    ⊢ Iff (LT.lt (MeasureTheory.lintegral (μ.restrict s) fun a => ENorm.enorm (f a …
  -/
  rw [hs.setLIntegral_eq ht]
  /-
    case hf
    G : Type u_1
    α : Type u_3
    E : Type u_5
    inst✝⁷ : Group G
    inst✝⁶ : MulAction G α
    inst✝⁵ : MeasurableSpace α
    inst✝⁴ : NormedAddCommGroup E
    s t : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    ht : MeasureTheory.IsFundamentalDomain G t μ
    f : α → E
    hf : ∀ (g : G) (x : α), Eq (f (HSMul.hSMul g x)) (f x)
    ⊢ ∀ (g : G) (x : α), Eq (ENorm.enorm (f (HSMul.hSMul g x))) (ENorm.enorm (f x))
  -/
  intro g x; rw [hf]
             /-
               🎉 no goals
             -/


@[to_additive]
protected theorem integrableOn_iff (hs : IsFundamentalDomain G s μ) (ht : IsFundamentalDomain G t μ)
    {f : α → E} (hf : ∀ (g : G) (x), f (g • x) = f x) : IntegrableOn f s μ ↔ IntegrableOn f t μ :=
  and_congr (hs.aEStronglyMeasurable_on_iff ht hf) (hs.hasFiniteIntegral_on_iff ht hf)


@[to_additive]
theorem integral_eq_tsum_of_ac (h : IsFundamentalDomain G s μ) (hν : ν ≪ μ) (f : α → E)
    (hf : Integrable f ν) : ∫ x, f x ∂ν = ∑' g : G, ∫ x in g • s, f x ∂ν := by
  /-
    G : Type u_1
    α : Type u_3
    E : Type u_5
    inst✝⁸ : Group G
    inst✝⁷ : MulAction G α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : NormedAddCommGroup E
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝⁴ : MeasurableSpace G
    inst✝³ : MeasurableSMul G α
    inst✝² : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝¹ : Countable G
    ν : MeasureTheory.Measure α
    inst✝ : NormedSpace Real E
    h : MeasureTheory.IsFundamentalDomain G s μ
    hν : ν.AbsolutelyContinuous μ
    f : α → E
    hf : MeasureTheory.Integrable f ν
    ⊢ Eq (MeasureTheory.integral ν fun x => f x) (tsum fun g => MeasureTheory.inte …
  -/
  rw [← MeasureTheory.integral_sum_measure, h.sum_restrict_of_ac hν]
  /-
    G : Type u_1
    α : Type u_3
    E : Type u_5
    inst✝⁸ : Group G
    inst✝⁷ : MulAction G α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : NormedAddCommGroup E
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝⁴ : MeasurableSpace G
    inst✝³ : MeasurableSMul G α
    inst✝² : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝¹ : Countable G
    ν : MeasureTheory.Measure α
    inst✝ : NormedSpace Real E
    h : MeasureTheory.IsFundamentalDomain G s μ
    hν : ν.AbsolutelyContinuous μ
    f : α → E
    hf : MeasureTheory.Integrable f ν
    ⊢ MeasureTheory.Integrable f (MeasureTheory.Measure.sum fun g => ν.restrict (H …
  -/
  rw [h.sum_restrict_of_ac hν]
  /-
    G : Type u_1
    α : Type u_3
    E : Type u_5
    inst✝⁸ : Group G
    inst✝⁷ : MulAction G α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : NormedAddCommGroup E
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝⁴ : MeasurableSpace G
    inst✝³ : MeasurableSMul G α
    inst✝² : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝¹ : Countable G
    ν : MeasureTheory.Measure α
    inst✝ : NormedSpace Real E
    h : MeasureTheory.IsFundamentalDomain G s μ
    hν : ν.AbsolutelyContinuous μ
    f : α → E
    hf : MeasureTheory.Integrable f ν
    ⊢ MeasureTheory.Integrable f ν
  -/
  exact hf
  /-
    🎉 no goals
  -/


@[to_additive]
theorem integral_eq_tsum (h : IsFundamentalDomain G s μ) (f : α → E) (hf : Integrable f μ) :
    ∫ x, f x ∂μ = ∑' g : G, ∫ x in g • s, f x ∂μ :=
                               /-
                                 G : Type u_1
                                 α : Type u_3
                                 E : Type u_5
                                 inst✝⁸ : Group G
                                 inst✝⁷ : MulAction G α
                                 inst✝⁶ : MeasurableSpace α
                                 inst✝⁵ : NormedAddCommGroup E
                                 s : Set α
                                 μ : MeasureTheory.Measure α
                                 inst✝⁴ : MeasurableSpace G
                                 inst✝³ : MeasurableSMul G α
                                 inst✝² : MeasureTheory.SMulInvariantMeasure G α μ
                                 inst✝¹ : Countable G
                                 inst✝ : NormedSpace Real E
                                 h : MeasureTheory.IsFundamentalDomain G s μ
                                 f : α → E
                                 hf : MeasureTheory.Integrable f μ
                                 ⊢ μ.AbsolutelyContinuous μ
                               -/
  integral_eq_tsum_of_ac h (by rfl) f hf
                               /-
                                 🎉 no goals
                               -/


@[to_additive]
theorem integral_eq_tsum' (h : IsFundamentalDomain G s μ) (f : α → E) (hf : Integrable f μ) :
    ∫ x, f x ∂μ = ∑' g : G, ∫ x in s, f (g⁻¹ • x) ∂μ :=
  calc
    ∫ x, f x ∂μ = ∑' g : G, ∫ x in g • s, f x ∂μ := h.integral_eq_tsum f hf
    _ = ∑' g : G, ∫ x in g⁻¹ • s, f x ∂μ := ((Equiv.inv G).tsum_eq _).symm
    _ = ∑' g : G, ∫ x in s, f (g⁻¹ • x) ∂μ := tsum_congr fun g =>
      (measurePreserving_smul g⁻¹ μ).setIntegral_image_emb (measurableEmbedding_const_smul _) _ _


@[to_additive] lemma integral_eq_tsum'' (h : IsFundamentalDomain G s μ)
    (f : α → E) (hf : Integrable f μ) : ∫ x, f x ∂μ = ∑' g : G, ∫ x in s, f (g • x) ∂μ :=
  (integral_eq_tsum' h f hf).trans ((Equiv.inv G).tsum_eq (fun g ↦ ∫ (x : α) in s, f (g • x) ∂μ))


@[to_additive]
theorem setIntegral_eq_tsum (h : IsFundamentalDomain G s μ) {f : α → E} {t : Set α}
    (hf : IntegrableOn f t μ) : ∫ x in t, f x ∂μ = ∑' g : G, ∫ x in t ∩ g • s, f x ∂μ :=
  calc
    ∫ x in t, f x ∂μ = ∑' g : G, ∫ x in g • s, f x ∂μ.restrict t :=
      h.integral_eq_tsum_of_ac restrict_le_self.absolutelyContinuous f hf
    _ = ∑' g : G, ∫ x in t ∩ g • s, f x ∂μ := by
      /-
        G : Type u_1
        α : Type u_3
        E : Type u_5
        inst✝⁸ : Group G
        inst✝⁷ : MulAction G α
        inst✝⁶ : MeasurableSpace α
        inst✝⁵ : NormedAddCommGroup E
        s : Set α
        μ : MeasureTheory.Measure α
        inst✝⁴ : MeasurableSpace G
        inst✝³ : MeasurableSMul G α
        inst✝² : MeasureTheory.SMulInvariantMeasure G α μ
        inst✝¹ : Countable G
        inst✝ : NormedSpace Real E
        h : MeasureTheory.IsFundamentalDomain G s μ
        f : α → E
        t : Set α
        hf : MeasureTheory.IntegrableOn f t μ
        ⊢ Eq (tsum fun g => MeasureTheory.integral ((μ.restrict t).restrict (HSMul.hSM …
      -/
      simp only [h.restrict_restrict, measure_smul, inter_comm]
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-04-17")]
alias set_integral_eq_tsum := setIntegral_eq_tsum


@[to_additive]
theorem setIntegral_eq_tsum' (h : IsFundamentalDomain G s μ) {f : α → E} {t : Set α}
    (hf : IntegrableOn f t μ) : ∫ x in t, f x ∂μ = ∑' g : G, ∫ x in g • t ∩ s, f (g⁻¹ • x) ∂μ :=
  calc
    ∫ x in t, f x ∂μ = ∑' g : G, ∫ x in t ∩ g • s, f x ∂μ := h.setIntegral_eq_tsum hf
    _ = ∑' g : G, ∫ x in t ∩ g⁻¹ • s, f x ∂μ := ((Equiv.inv G).tsum_eq _).symm
                                                         /-
                                                           G : Type u_1
                                                           α : Type u_3
                                                           E : Type u_5
                                                           inst✝⁸ : Group G
                                                           inst✝⁷ : MulAction G α
                                                           inst✝⁶ : MeasurableSpace α
                                                           inst✝⁵ : NormedAddCommGroup E
                                                           s : Set α
                                                           μ : MeasureTheory.Measure α
                                                           inst✝⁴ : MeasurableSpace G
                                                           inst✝³ : MeasurableSMul G α
                                                           inst✝² : MeasureTheory.SMulInvariantMeasure G α μ
                                                           inst✝¹ : Countable G
                                                           inst✝ : NormedSpace Real E
                                                           h : MeasureTheory.IsFundamentalDomain G s μ
                                                           f : α → E
                                                           t : Set α
                                                           hf : MeasureTheory.IntegrableOn f t μ
                                                           ⊢ Eq (tsum fun g => MeasureTheory.integral (μ.restrict (Inter.inter t (HSMul.h …
                                                         -/
    _ = ∑' g : G, ∫ x in g⁻¹ • (g • t ∩ s), f x ∂μ := by simp only [smul_set_inter, inv_smul_smul]
                                                         /-
                                                           🎉 no goals
                                                         -/
    _ = ∑' g : G, ∫ x in g • t ∩ s, f (g⁻¹ • x) ∂μ :=
      tsum_congr fun g =>
        (measurePreserving_smul g⁻¹ μ).setIntegral_image_emb (measurableEmbedding_const_smul _) _ _


@[deprecated (since := "2024-04-17")]
alias set_integral_eq_tsum' := setIntegral_eq_tsum'


@[to_additive]
protected theorem setIntegral_eq (hs : IsFundamentalDomain G s μ) (ht : IsFundamentalDomain G t μ)
    {f : α → E} (hf : ∀ (g : G) (x), f (g • x) = f x) : ∫ x in s, f x ∂μ = ∫ x in t, f x ∂μ := by
  /-
    G : Type u_1
    α : Type u_3
    E : Type u_5
    inst✝⁸ : Group G
    inst✝⁷ : MulAction G α
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : NormedAddCommGroup E
    s t : Set α
    μ : MeasureTheory.Measure α
    inst✝⁴ : MeasurableSpace G
    inst✝³ : MeasurableSMul G α
    inst✝² : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝¹ : Countable G
    inst✝ : NormedSpace Real E
    hs : MeasureTheory.IsFundamentalDomain G s μ
    ht : MeasureTheory.IsFundamentalDomain G t μ
    f : α → E
    hf : ∀ (g : G) (x : α), Eq (f (HSMul.hSMul g x)) (f x)
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => f x) (MeasureTheory.integ …
  -/
  by_cases hfs : IntegrableOn f s μ
    /-
      case pos
      G : Type u_1
      α : Type u_3
      E : Type u_5
      inst✝⁸ : Group G
      inst✝⁷ : MulAction G α
      inst✝⁶ : MeasurableSpace α
      inst✝⁵ : NormedAddCommGroup E
      s t : Set α
      μ : MeasureTheory.Measure α
      inst✝⁴ : MeasurableSpace G
      inst✝³ : MeasurableSMul G α
      inst✝² : MeasureTheory.SMulInvariantMeasure G α μ
      inst✝¹ : Countable G
      inst✝ : NormedSpace Real E
      hs : MeasureTheory.IsFundamentalDomain G s μ
      ht : MeasureTheory.IsFundamentalDomain G t μ
      f : α → E
      hf : ∀ (g : G) (x : α), Eq (f (HSMul.hSMul g x)) (f x)
      hfs : MeasureTheory.IntegrableOn f s μ
      ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => f x) (MeasureTheory.integ …
    -/
  · have hft : IntegrableOn f t μ := by rwa [ht.integrableOn_iff hs hf]
    calc
      ∫ x in s, f x ∂μ = ∑' g : G, ∫ x in s ∩ g • t, f x ∂μ := ht.setIntegral_eq_tsum hfs
      _ = ∑' g : G, ∫ x in g • t ∩ s, f (g⁻¹ • x) ∂μ := by simp only [hf, inter_comm]
      _ = ∫ x in t, f x ∂μ := (hs.setIntegral_eq_tsum' hft).symm
    /-
      case neg
      G : Type u_1
      α : Type u_3
      E : Type u_5
      inst✝⁸ : Group G
      inst✝⁷ : MulAction G α
      inst✝⁶ : MeasurableSpace α
      inst✝⁵ : NormedAddCommGroup E
      s t : Set α
      μ : MeasureTheory.Measure α
      inst✝⁴ : MeasurableSpace G
      inst✝³ : MeasurableSMul G α
      inst✝² : MeasureTheory.SMulInvariantMeasure G α μ
      inst✝¹ : Countable G
      inst✝ : NormedSpace Real E
      hs : MeasureTheory.IsFundamentalDomain G s μ
      ht : MeasureTheory.IsFundamentalDomain G t μ
      f : α → E
      hf : ∀ (g : G) (x : α), Eq (f (HSMul.hSMul g x)) (f x)
      hfs : Not (MeasureTheory.IntegrableOn f s μ)
      ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => f x) (MeasureTheory.integ …
    -/
  · rw [integral_undef hfs, integral_undef]
    /-
      case neg
      G : Type u_1
      α : Type u_3
      E : Type u_5
      inst✝⁸ : Group G
      inst✝⁷ : MulAction G α
      inst✝⁶ : MeasurableSpace α
      inst✝⁵ : NormedAddCommGroup E
      s t : Set α
      μ : MeasureTheory.Measure α
      inst✝⁴ : MeasurableSpace G
      inst✝³ : MeasurableSMul G α
      inst✝² : MeasureTheory.SMulInvariantMeasure G α μ
      inst✝¹ : Countable G
      inst✝ : NormedSpace Real E
      hs : MeasureTheory.IsFundamentalDomain G s μ
      ht : MeasureTheory.IsFundamentalDomain G t μ
      f : α → E
      hf : ∀ (g : G) (x : α), Eq (f (HSMul.hSMul g x)) (f x)
      hfs : Not (MeasureTheory.IntegrableOn f s μ)
      ⊢ Not (MeasureTheory.Integrable f (μ.restrict t))
    -/
    rwa [hs.integrableOn_iff ht hf] at hfs
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias set_integral_eq := MeasureTheory.IsFundamentalDomain.setIntegral_eq


/-- If the action of a countable group `G` admits an invariant measure `μ` with a fundamental domain
`s`, then every null-measurable set `t` such that the sets `g • t ∩ s` are pairwise a.e.-disjoint
has measure at most `μ s`. -/
@[to_additive "If the additive action of a countable group `G` admits an invariant measure `μ` with
  a fundamental domain `s`, then every null-measurable set `t` such that the sets `g +ᵥ t ∩ s` are
  pairwise a.e.-disjoint has measure at most `μ s`."]
theorem measure_le_of_pairwise_disjoint (hs : IsFundamentalDomain G s μ)
    (ht : NullMeasurableSet t μ) (hd : Pairwise (AEDisjoint μ on fun g : G => g • t ∩ s)) :
    μ t ≤ μ s :=
  calc
    μ t = ∑' g : G, μ (g • t ∩ s) := hs.measure_eq_tsum t
    _ = μ (⋃ g : G, g • t ∩ s) := Eq.symm <| measure_iUnion₀ hd fun _ =>
      (ht.smul _).inter hs.nullMeasurableSet
    _ ≤ μ s := measure_mono (iUnion_subset fun _ => inter_subset_right)


/-- If the action of a countable group `G` admits an invariant measure `μ` with a fundamental domain
`s`, then every null-measurable set `t` of measure strictly greater than `μ s` contains two
points `x y` such that `g • x = y` for some `g ≠ 1`. -/
@[to_additive "If the additive action of a countable group `G` admits an invariant measure `μ` with
  a fundamental domain `s`, then every null-measurable set `t` of measure strictly greater than
  `μ s` contains two points `x y` such that `g +ᵥ x = y` for some `g ≠ 0`."]
theorem exists_ne_one_smul_eq (hs : IsFundamentalDomain G s μ) (htm : NullMeasurableSet t μ)
    (ht : μ s < μ t) : ∃ x ∈ t, ∃ y ∈ t, ∃ g, g ≠ (1 : G) ∧ g • x = y := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s t : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    htm : MeasureTheory.NullMeasurableSet t μ
    ht : LT.lt (μ s) (μ t)
    ⊢ Exists fun x => And (Membership.mem t x) (Exists fun y => And (Membership.me …
  -/
  contrapose! ht
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s t : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    htm : MeasureTheory.NullMeasurableSet t μ
    ht : ∀ (x : α), Membership.mem t x → ∀ (y : α), Membership.mem t y → ∀ (g : G) …
    ⊢ LE.le (μ t) (μ s)
  -/
  refine hs.measure_le_of_pairwise_disjoint htm (Pairwise.aedisjoint fun g₁ g₂ hne => ?_)
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s t : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    htm : MeasureTheory.NullMeasurableSet t μ
    ht : ∀ (x : α), Membership.mem t x → ∀ (y : α), Membership.mem t y → ∀ (g : G) …
    g₁ g₂ : G
    hne : Ne g₁ g₂
    ⊢ Function.onFun Disjoint (fun g => Inter.inter (HSMul.hSMul g t) s) g₁ g₂
  -/
  dsimp [Function.onFun]
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s t : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    htm : MeasureTheory.NullMeasurableSet t μ
    ht : ∀ (x : α), Membership.mem t x → ∀ (y : α), Membership.mem t y → ∀ (g : G) …
    g₁ g₂ : G
    hne : Ne g₁ g₂
    ⊢ Disjoint (Inter.inter (HSMul.hSMul g₁ t) s) (Inter.inter (HSMul.hSMul g₂ t) s)
  -/
  refine (Disjoint.inf_left _ ?_).inf_right _
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s t : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    htm : MeasureTheory.NullMeasurableSet t μ
    ht : ∀ (x : α), Membership.mem t x → ∀ (y : α), Membership.mem t y → ∀ (g : G) …
    g₁ g₂ : G
    hne : Ne g₁ g₂
    ⊢ Disjoint (HSMul.hSMul g₁ t) (HSMul.hSMul g₂ t)
  -/
  rw [Set.disjoint_left]
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s t : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    htm : MeasureTheory.NullMeasurableSet t μ
    ht : ∀ (x : α), Membership.mem t x → ∀ (y : α), Membership.mem t y → ∀ (g : G) …
    g₁ g₂ : G
    hne : Ne g₁ g₂
    ⊢ ∀ ⦃a : α⦄, Membership.mem (HSMul.hSMul g₁ t) a → Not (Membership.mem (HSMul. …
  -/
  rintro _ ⟨x, hx, rfl⟩ ⟨y, hy, hxy : g₂ • y = g₁ • x⟩
  /-
    case intro.intro.intro.intro
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s t : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    htm : MeasureTheory.NullMeasurableSet t μ
    ht : ∀ (x : α), Membership.mem t x → ∀ (y : α), Membership.mem t y → ∀ (g : G) …
    g₁ g₂ : G
    hne : Ne g₁ g₂
    x : α
    hx : Membership.mem t x
    y : α
    hy : Membership.mem t y
    hxy : Eq (HSMul.hSMul g₂ y) (HSMul.hSMul g₁ x)
    ⊢ False
  -/
  refine ht x hx y hy (g₂⁻¹ * g₁) (mt inv_mul_eq_one.1 hne.symm) ?_
  /-
    case intro.intro.intro.intro
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s t : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    htm : MeasureTheory.NullMeasurableSet t μ
    ht : ∀ (x : α), Membership.mem t x → ∀ (y : α), Membership.mem t y → ∀ (g : G) …
    g₁ g₂ : G
    hne : Ne g₁ g₂
    x : α
    hx : Membership.mem t x
    y : α
    hy : Membership.mem t y
    hxy : Eq (HSMul.hSMul g₂ y) (HSMul.hSMul g₁ x)
    ⊢ Eq (HSMul.hSMul (HMul.hMul (Inv.inv g₂) g₁) x) y
  -/
  rw [mul_smul, ← hxy, inv_smul_smul]
  /-
    🎉 no goals
  -/


/-- If `f` is invariant under the action of a countable group `G`, and `μ` is a `G`-invariant
  measure with a fundamental domain `s`, then the `essSup` of `f` restricted to `s` is the same as
  that of `f` on all of its domain. -/
@[to_additive "If `f` is invariant under the action of a countable additive group `G`, and `μ` is a
  `G`-invariant measure with a fundamental domain `s`, then the `essSup` of `f` restricted to `s`
  is the same as that of `f` on all of its domain."]
theorem essSup_measure_restrict (hs : IsFundamentalDomain G s μ) {f : α → ℝ≥0∞}
    (hf : ∀ γ : G, ∀ x : α, f (γ • x) = f x) : essSup f (μ.restrict s) = essSup f μ := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    f : α → ENNReal
    hf : ∀ (γ : G) (x : α), Eq (f (HSMul.hSMul γ x)) (f x)
    ⊢ Eq (essSup f (μ.restrict s)) (essSup f μ)
  -/
  refine le_antisymm (essSup_mono_measure' Measure.restrict_le_self) ?_
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    f : α → ENNReal
    hf : ∀ (γ : G) (x : α), Eq (f (HSMul.hSMul γ x)) (f x)
    ⊢ LE.le (essSup f μ) (essSup f (μ.restrict s))
  -/
  rw [essSup_eq_sInf (μ.restrict s) f, essSup_eq_sInf μ f]
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    f : α → ENNReal
    hf : ∀ (γ : G) (x : α), Eq (f (HSMul.hSMul γ x)) (f x)
    ⊢ LE.le (InfSet.sInf (setOf fun a => Eq (μ (setOf fun x => LT.lt a (f x))) 0)) …
  -/
  refine sInf_le_sInf ?_
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    f : α → ENNReal
    hf : ∀ (γ : G) (x : α), Eq (f (HSMul.hSMul γ x)) (f x)
    ⊢ HasSubset.Subset (setOf fun a => Eq ((μ.restrict s) (setOf fun x => LT.lt a  …
  -/
  rintro a (ha : (μ.restrict s) {x : α | a < f x} = 0)
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    f : α → ENNReal
    hf : ∀ (γ : G) (x : α), Eq (f (HSMul.hSMul γ x)) (f x)
    a : ENNReal
    ha : Eq ((μ.restrict s) (setOf fun x => LT.lt a (f x))) 0
    ⊢ Membership.mem (setOf fun a => Eq (μ (setOf fun x => LT.lt a (f x))) 0) a
  -/
  rw [Measure.restrict_apply₀' hs.nullMeasurableSet] at ha
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    f : α → ENNReal
    hf : ∀ (γ : G) (x : α), Eq (f (HSMul.hSMul γ x)) (f x)
    a : ENNReal
    ha : Eq (μ (Inter.inter (setOf fun x => LT.lt a (f x)) s)) 0
    ⊢ Membership.mem (setOf fun a => Eq (μ (setOf fun x => LT.lt a (f x))) 0) a
  -/
  refine measure_zero_of_invariant hs _ ?_ ha
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    f : α → ENNReal
    hf : ∀ (γ : G) (x : α), Eq (f (HSMul.hSMul γ x)) (f x)
    a : ENNReal
    ha : Eq (μ (Inter.inter (setOf fun x => LT.lt a (f x)) s)) 0
    ⊢ ∀ (g : G), Eq (HSMul.hSMul g (setOf fun x => LT.lt a (f x))) (setOf fun x => …
  -/
  intro γ
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    f : α → ENNReal
    hf : ∀ (γ : G) (x : α), Eq (f (HSMul.hSMul γ x)) (f x)
    a : ENNReal
    ha : Eq (μ (Inter.inter (setOf fun x => LT.lt a (f x)) s)) 0
    γ : G
    ⊢ Eq (HSMul.hSMul γ (setOf fun x => LT.lt a (f x))) (setOf fun x => LT.lt a (f …
  -/
  ext x
  /-
    case h
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    f : α → ENNReal
    hf : ∀ (γ : G) (x : α), Eq (f (HSMul.hSMul γ x)) (f x)
    a : ENNReal
    ha : Eq (μ (Inter.inter (setOf fun x => LT.lt a (f x)) s)) 0
    γ : G
    x : α
    ⊢ Iff (Membership.mem (HSMul.hSMul γ (setOf fun x => LT.lt a (f x))) x) (Membe …
  -/
  rw [mem_smul_set_iff_inv_smul_mem]
  /-
    case h
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    s : Set α
    μ : MeasureTheory.Measure α
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : Countable G
    hs : MeasureTheory.IsFundamentalDomain G s μ
    f : α → ENNReal
    hf : ∀ (γ : G) (x : α), Eq (f (HSMul.hSMul γ x)) (f x)
    a : ENNReal
    ha : Eq (μ (Inter.inter (setOf fun x => LT.lt a (f x)) s)) 0
    γ : G
    x : α
    ⊢ Iff (Membership.mem (setOf fun x => LT.lt a (f x)) (HSMul.hSMul (Inv.inv γ)  …
  -/
  simp only [mem_setOf_eq, hf γ⁻¹ x]
  /-
    🎉 no goals
  -/


/-- The boundary of a fundamental domain, those points of the domain that also lie in a nontrivial
translate. -/
@[to_additive MeasureTheory.addFundamentalFrontier "The boundary of a fundamental domain, those
  points of the domain that also lie in a nontrivial translate."]
def fundamentalFrontier : Set α :=
  s ∩ ⋃ (g : G) (_ : g ≠ 1), g • s


/-- The interior of a fundamental domain, those points of the domain not lying in any translate. -/
@[to_additive MeasureTheory.addFundamentalInterior "The interior of a fundamental domain, those
  points of the domain not lying in any translate."]
def fundamentalInterior : Set α :=
  s \ ⋃ (g : G) (_ : g ≠ 1), g • s


@[to_additive (attr := simp) MeasureTheory.mem_addFundamentalFrontier]
theorem mem_fundamentalFrontier :
    x ∈ fundamentalFrontier G s ↔ x ∈ s ∧ ∃ g : G, g ≠ 1 ∧ x ∈ g • s := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝¹ : Group G
    inst✝ : MulAction G α
    s : Set α
    x : α
    ⊢ Iff (Membership.mem (MeasureTheory.fundamentalFrontier G s) x) (And (Members …
  -/
  simp [fundamentalFrontier]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) MeasureTheory.mem_addFundamentalInterior]
theorem mem_fundamentalInterior :
    x ∈ fundamentalInterior G s ↔ x ∈ s ∧ ∀ g : G, g ≠ 1 → x ∉ g • s := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝¹ : Group G
    inst✝ : MulAction G α
    s : Set α
    x : α
    ⊢ Iff (Membership.mem (MeasureTheory.fundamentalInterior G s) x) (And (Members …
  -/
  simp [fundamentalInterior]
  /-
    🎉 no goals
  -/


@[to_additive MeasureTheory.addFundamentalFrontier_subset]
theorem fundamentalFrontier_subset : fundamentalFrontier G s ⊆ s :=
  inter_subset_left


@[to_additive MeasureTheory.addFundamentalInterior_subset]
theorem fundamentalInterior_subset : fundamentalInterior G s ⊆ s :=
  diff_subset


@[to_additive MeasureTheory.disjoint_addFundamentalInterior_addFundamentalFrontier]
theorem disjoint_fundamentalInterior_fundamentalFrontier :
    Disjoint (fundamentalInterior G s) (fundamentalFrontier G s) :=
  disjoint_sdiff_self_left.mono_right inf_le_right


@[to_additive (attr := simp) MeasureTheory.addFundamentalInterior_union_addFundamentalFrontier]
theorem fundamentalInterior_union_fundamentalFrontier :
    fundamentalInterior G s ∪ fundamentalFrontier G s = s :=
  diff_union_inter _ _


@[to_additive (attr := simp) MeasureTheory.addFundamentalFrontier_union_addFundamentalInterior]
theorem fundamentalFrontier_union_fundamentalInterior :
    fundamentalFrontier G s ∪ fundamentalInterior G s = s :=
  inter_union_diff _ _


@[to_additive (attr := simp) MeasureTheory.sdiff_addFundamentalInterior]
theorem sdiff_fundamentalInterior : s \ fundamentalInterior G s = fundamentalFrontier G s :=
  sdiff_sdiff_right_self


@[to_additive (attr := simp) MeasureTheory.sdiff_addFundamentalFrontier]
theorem sdiff_fundamentalFrontier : s \ fundamentalFrontier G s = fundamentalInterior G s :=
  diff_self_inter


@[to_additive (attr := simp) MeasureTheory.addFundamentalFrontier_vadd]
theorem fundamentalFrontier_smul [Group H] [MulAction H α] [SMulCommClass H G α] (g : H) :
    fundamentalFrontier G (g • s) = g • fundamentalFrontier G s := by
  /-
    G : Type u_1
    H : Type u_2
    α : Type u_3
    inst✝⁴ : Group G
    inst✝³ : MulAction G α
    s : Set α
    inst✝² : Group H
    inst✝¹ : MulAction H α
    inst✝ : SMulCommClass H G α
    g : H
    ⊢ Eq (MeasureTheory.fundamentalFrontier G (HSMul.hSMul g s)) (HSMul.hSMul g (M …
  -/
  simp_rw [fundamentalFrontier, smul_set_inter, smul_set_iUnion, smul_comm g (_ : G) (_ : Set α)]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) MeasureTheory.addFundamentalInterior_vadd]
theorem fundamentalInterior_smul [Group H] [MulAction H α] [SMulCommClass H G α] (g : H) :
    fundamentalInterior G (g • s) = g • fundamentalInterior G s := by
  /-
    G : Type u_1
    H : Type u_2
    α : Type u_3
    inst✝⁴ : Group G
    inst✝³ : MulAction G α
    s : Set α
    inst✝² : Group H
    inst✝¹ : MulAction H α
    inst✝ : SMulCommClass H G α
    g : H
    ⊢ Eq (MeasureTheory.fundamentalInterior G (HSMul.hSMul g s)) (HSMul.hSMul g (M …
  -/
  simp_rw [fundamentalInterior, smul_set_sdiff, smul_set_iUnion, smul_comm g (_ : G) (_ : Set α)]
  /-
    🎉 no goals
  -/


@[to_additive MeasureTheory.pairwise_disjoint_addFundamentalInterior]
theorem pairwise_disjoint_fundamentalInterior :
    Pairwise (Disjoint on fun g : G => g • fundamentalInterior G s) := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝¹ : Group G
    inst✝ : MulAction G α
    s : Set α
    ⊢ Pairwise (Function.onFun Disjoint fun g => HSMul.hSMul g (MeasureTheory.fund …
  -/
  refine fun a b hab => disjoint_left.2 ?_
  /-
    G : Type u_1
    α : Type u_3
    inst✝¹ : Group G
    inst✝ : MulAction G α
    s : Set α
    a b : G
    hab : Ne a b
    ⊢ ∀ ⦃a_1 : α⦄, Membership.mem ((fun g => HSMul.hSMul g (MeasureTheory.fundamen …
  -/
  rintro _ ⟨x, hx, rfl⟩ ⟨y, hy, hxy⟩
  /-
    case intro.intro.intro.intro
    G : Type u_1
    α : Type u_3
    inst✝¹ : Group G
    inst✝ : MulAction G α
    s : Set α
    a b : G
    hab : Ne a b
    x : α
    hx : Membership.mem (MeasureTheory.fundamentalInterior G s) x
    y : α
    hy : Membership.mem (MeasureTheory.fundamentalInterior G s) y
    hxy : Eq ((fun x => HSMul.hSMul b x) y) ((fun x => HSMul.hSMul a x) x)
    ⊢ False
  -/
  rw [mem_fundamentalInterior] at hx hy
  /-
    case intro.intro.intro.intro
    G : Type u_1
    α : Type u_3
    inst✝¹ : Group G
    inst✝ : MulAction G α
    s : Set α
    a b : G
    hab : Ne a b
    x : α
    hx : And (Membership.mem s x) (∀ (g : G), Ne g 1 → Not (Membership.mem (HSMul. …
    y : α
    hy : And (Membership.mem s y) (∀ (g : G), Ne g 1 → Not (Membership.mem (HSMul. …
    hxy : Eq ((fun x => HSMul.hSMul b x) y) ((fun x => HSMul.hSMul a x) x)
    ⊢ False
  -/
  refine hx.2 (a⁻¹ * b) ?_ ?_
    /-
      case intro.intro.intro.intro.refine_1
      G : Type u_1
      α : Type u_3
      inst✝¹ : Group G
      inst✝ : MulAction G α
      s : Set α
      a b : G
      hab : Ne a b
      x : α
      hx : And (Membership.mem s x) (∀ (g : G), Ne g 1 → Not (Membership.mem (HSMul. …
      y : α
      hy : And (Membership.mem s y) (∀ (g : G), Ne g 1 → Not (Membership.mem (HSMul. …
      hxy : Eq ((fun x => HSMul.hSMul b x) y) ((fun x => HSMul.hSMul a x) x)
      ⊢ Ne (HMul.hMul (Inv.inv a) b) 1
    -/
  · rwa [Ne, inv_mul_eq_iff_eq_mul, mul_one, eq_comm]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      G : Type u_1
      α : Type u_3
      inst✝¹ : Group G
      inst✝ : MulAction G α
      s : Set α
      a b : G
      hab : Ne a b
      x : α
      hx : And (Membership.mem s x) (∀ (g : G), Ne g 1 → Not (Membership.mem (HSMul. …
      y : α
      hy : And (Membership.mem s y) (∀ (g : G), Ne g 1 → Not (Membership.mem (HSMul. …
      hxy : Eq ((fun x => HSMul.hSMul b x) y) ((fun x => HSMul.hSMul a x) x)
      ⊢ Membership.mem (HSMul.hSMul (HMul.hMul (Inv.inv a) b) s) x
    -/
  · simpa [mul_smul, ← hxy, mem_inv_smul_set_iff] using hy.1
    /-
      🎉 no goals
    -/


@[to_additive MeasureTheory.NullMeasurableSet.addFundamentalFrontier]
protected theorem NullMeasurableSet.fundamentalFrontier (hs : NullMeasurableSet s μ) :
    NullMeasurableSet (fundamentalFrontier G s) μ :=
  hs.inter <| .iUnion fun _ => .iUnion fun _ => hs.smul _


@[to_additive MeasureTheory.NullMeasurableSet.addFundamentalInterior]
protected theorem NullMeasurableSet.fundamentalInterior (hs : NullMeasurableSet s μ) :
    NullMeasurableSet (fundamentalInterior G s) μ :=
  hs.diff <| .iUnion fun _ => .iUnion fun _ => hs.smul _


@[to_additive MeasureTheory.IsAddFundamentalDomain.measure_addFundamentalFrontier]
theorem measure_fundamentalFrontier : μ (fundamentalFrontier G s) = 0 := by
  simpa only [fundamentalFrontier, iUnion₂_inter, one_smul, measure_iUnion_null_iff, inter_comm s,
    Function.onFun] using fun g (hg : g ≠ 1) => hs.aedisjoint hg


@[to_additive MeasureTheory.IsAddFundamentalDomain.measure_addFundamentalInterior]
theorem measure_fundamentalInterior : μ (fundamentalInterior G s) = μ s :=
  measure_diff_null' hs.measure_fundamentalFrontier


protected theorem fundamentalInterior : IsFundamentalDomain G (fundamentalInterior G s) μ where
  nullMeasurableSet := hs.nullMeasurableSet.fundamentalInterior _ _
  ae_covers := by
    simp_rw [ae_iff, not_exists, ← mem_inv_smul_set_iff, setOf_forall, ← compl_setOf,
      setOf_mem_eq, ← compl_iUnion]
    have :
      ((⋃ g : G, g⁻¹ • s) \ ⋃ g : G, g⁻¹ • fundamentalFrontier G s) ⊆
        ⋃ g : G, g⁻¹ • fundamentalInterior G s := by
      simp_rw [diff_subset_iff, ← iUnion_union_distrib, ← smul_set_union (α := G) (β := α),
        fundamentalFrontier_union_fundamentalInterior]; rfl
    /-
      G : Type u_1
      α : Type u_3
      inst✝⁶ : Countable G
      inst✝⁵ : Group G
      inst✝⁴ : MulAction G α
      inst✝³ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hs : MeasureTheory.IsFundamentalDomain G s μ
      inst✝² : MeasurableSpace G
      inst✝¹ : MeasurableSMul G α
      inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
      this : HasSubset.Subset (SDiff.sdiff (Set.iUnion fun g => HSMul.hSMul (Inv.inv …
      ⊢ Eq (μ (HasCompl.compl (Set.iUnion fun i => HSMul.hSMul (Inv.inv i) (MeasureT …
    -/
    refine eq_bot_mono (μ.mono <| compl_subset_compl.2 this) ?_
    simp only [iUnion_inv_smul, compl_sdiff, ENNReal.bot_eq_zero, himp_eq, sup_eq_union,
      @iUnion_smul_eq_setOf_exists _ _ _ _ s]
    exact measure_union_null
      (measure_iUnion_null fun _ => measure_smul_null hs.measure_fundamentalFrontier _) hs.ae_covers
  aedisjoint := (pairwise_disjoint_fundamentalInterior _ _).mono fun _ _ => Disjoint.aedisjoint


local notation "α_mod_G" => MulAction.orbitRel G α


local notation "π" => @Quotient.mk _ α_mod_G


@[to_additive addMeasure_map_restrict_apply]
lemma measure_map_restrict_apply (s : Set α) {U : Set (Quotient α_mod_G)}
    (meas_U : MeasurableSet U) :
    (μ.restrict s).map π U = μ ((π ⁻¹' U) ∩ s) := by
  rw [map_apply (f := π) (fun V hV ↦ measurableSet_quotient.mp hV) meas_U,
    Measure.restrict_apply (t := (Quotient.mk α_mod_G ⁻¹' U)) (measurableSet_quotient.mp meas_U)]


@[to_additive]
lemma IsFundamentalDomain.quotientMeasure_eq [Countable G] [MeasurableSpace G] {s t : Set α}
    [SMulInvariantMeasure G α μ] [MeasurableSMul G α] (fund_dom_s : IsFundamentalDomain G s μ)
    (fund_dom_t : IsFundamentalDomain G t μ) :
    (μ.restrict s).map π = (μ.restrict t).map π := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : Countable G
    inst✝² : MeasurableSpace G
    s t : Set α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : MeasurableSMul G α
    fund_dom_s : MeasureTheory.IsFundamentalDomain G s μ
    fund_dom_t : MeasureTheory.IsFundamentalDomain G t μ
    ⊢ Eq (MeasureTheory.Measure.map (Quotient.mk (MulAction.orbitRel G α)) (μ.rest …
  -/
  ext U meas_U
  /-
    case h
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : Countable G
    inst✝² : MeasurableSpace G
    s t : Set α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : MeasurableSMul G α
    fund_dom_s : MeasureTheory.IsFundamentalDomain G s μ
    fund_dom_t : MeasureTheory.IsFundamentalDomain G t μ
    U : Set (Quotient (MulAction.orbitRel G α))
    meas_U : MeasurableSet U
    ⊢ Eq ((MeasureTheory.Measure.map (Quotient.mk (MulAction.orbitRel G α)) (μ.res …
  -/
  rw [measure_map_restrict_apply (meas_U := meas_U), measure_map_restrict_apply (meas_U := meas_U)]
  /-
    case h
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : Countable G
    inst✝² : MeasurableSpace G
    s t : Set α
    inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
    inst✝ : MeasurableSMul G α
    fund_dom_s : MeasureTheory.IsFundamentalDomain G s μ
    fund_dom_t : MeasureTheory.IsFundamentalDomain G t μ
    U : Set (Quotient (MulAction.orbitRel G α))
    meas_U : MeasurableSet U
    ⊢ Eq (μ (Inter.inter (Set.preimage (Quotient.mk (MulAction.orbitRel G α)) U) s …
  -/
  apply MeasureTheory.IsFundamentalDomain.measure_set_eq fund_dom_s fund_dom_t
    /-
      case h.hA₀
      G : Type u_1
      α : Type u_3
      inst✝⁶ : Group G
      inst✝⁵ : MulAction G α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : Countable G
      inst✝² : MeasurableSpace G
      s t : Set α
      inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
      inst✝ : MeasurableSMul G α
      fund_dom_s : MeasureTheory.IsFundamentalDomain G s μ
      fund_dom_t : MeasureTheory.IsFundamentalDomain G t μ
      U : Set (Quotient (MulAction.orbitRel G α))
      meas_U : MeasurableSet U
      ⊢ MeasurableSet (Set.preimage (Quotient.mk (MulAction.orbitRel G α)) U)
    -/
  · exact measurableSet_quotient.mp meas_U
    /-
      🎉 no goals
    -/
    /-
      case h.hA
      G : Type u_1
      α : Type u_3
      inst✝⁶ : Group G
      inst✝⁵ : MulAction G α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : Countable G
      inst✝² : MeasurableSpace G
      s t : Set α
      inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
      inst✝ : MeasurableSMul G α
      fund_dom_s : MeasureTheory.IsFundamentalDomain G s μ
      fund_dom_t : MeasureTheory.IsFundamentalDomain G t μ
      U : Set (Quotient (MulAction.orbitRel G α))
      meas_U : MeasurableSet U
      ⊢ ∀ (g : G), Eq (Set.preimage (fun x => HSMul.hSMul g x) (Set.preimage (Quotie …
    -/
  · intro g
    /-
      case h.hA
      G : Type u_1
      α : Type u_3
      inst✝⁶ : Group G
      inst✝⁵ : MulAction G α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : Countable G
      inst✝² : MeasurableSpace G
      s t : Set α
      inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
      inst✝ : MeasurableSMul G α
      fund_dom_s : MeasureTheory.IsFundamentalDomain G s μ
      fund_dom_t : MeasureTheory.IsFundamentalDomain G t μ
      U : Set (Quotient (MulAction.orbitRel G α))
      meas_U : MeasurableSet U
      g : G
      ⊢ Eq (Set.preimage (fun x => HSMul.hSMul g x) (Set.preimage (Quotient.mk (MulA …
    -/
    ext x
    have : Quotient.mk α_mod_G (g • x) = Quotient.mk α_mod_G x := by
      apply Quotient.sound
      use g
    /-
      case h.hA.h
      G : Type u_1
      α : Type u_3
      inst✝⁶ : Group G
      inst✝⁵ : MulAction G α
      inst✝⁴ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : Countable G
      inst✝² : MeasurableSpace G
      s t : Set α
      inst✝¹ : MeasureTheory.SMulInvariantMeasure G α μ
      inst✝ : MeasurableSMul G α
      fund_dom_s : MeasureTheory.IsFundamentalDomain G s μ
      fund_dom_t : MeasureTheory.IsFundamentalDomain G t μ
      U : Set (Quotient (MulAction.orbitRel G α))
      meas_U : MeasurableSet U
      g : G
      x : α
      this : Eq (Quotient.mk (MulAction.orbitRel G α) (HSMul.hSMul g x)) (Quotient.m …
      ⊢ Iff (Membership.mem (Set.preimage (fun x => HSMul.hSMul g x) (Set.preimage ( …
    -/
    simp only [mem_preimage, this]
    /-
      🎉 no goals
    -/


/-- We say a quotient of `α` by `G` `HasAddFundamentalDomain` if there is a measurable set
  `s` for which `IsAddFundamentalDomain G s` holds. -/
class HasAddFundamentalDomain (G α : Type*) [Zero G] [VAdd G α] [MeasurableSpace α]
    (ν : Measure α := by volume_tac) : Prop where
  ExistsIsAddFundamentalDomain : ∃ s : Set α, IsAddFundamentalDomain G s ν


/-- We say a quotient of `α` by `G` `HasFundamentalDomain` if there is a measurable set `s` for
  which `IsFundamentalDomain G s` holds. -/
class HasFundamentalDomain (G : Type*) (α : Type*) [One G] [SMul G α] [MeasurableSpace α]
    (ν : Measure α := by volume_tac) : Prop where
  ExistsIsFundamentalDomain : ∃ (s : Set α), IsFundamentalDomain G s ν


open Classical in
/-- The `covolume` of an action of `G` on `α` the volume of some fundamental domain, or `0` if
none exists. -/
@[to_additive addCovolume "The `addCovolume` of an action of `G` on `α` is the volume of some
fundamental domain, or `0` if none exists."]
noncomputable def covolume (G α : Type*) [One G] [SMul G α] [MeasurableSpace α]
    (ν : Measure α := by volume_tac) : ℝ≥0∞ :=
  if funDom : HasFundamentalDomain G α ν then ν funDom.ExistsIsFundamentalDomain.choose else 0


/-- If there is a fundamental domain `s`, then `HasFundamentalDomain` holds. -/
@[to_additive]
lemma IsFundamentalDomain.hasFundamentalDomain (ν : Measure α) {s : Set α}
    (fund_dom_s : IsFundamentalDomain G s ν) :
    HasFundamentalDomain G α ν := ⟨⟨s, fund_dom_s⟩⟩


/-- The `covolume` can be computed by taking the `volume` of any given fundamental domain `s`. -/
@[to_additive]
lemma IsFundamentalDomain.covolume_eq_volume (ν : Measure α) [Countable G]
    [MeasurableSpace G] [MeasurableSMul G α] [SMulInvariantMeasure G α ν] {s : Set α}
    (fund_dom_s : IsFundamentalDomain G s ν) : covolume G α ν = ν s := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝³ : Countable G
    inst✝² : MeasurableSpace G
    inst✝¹ : MeasurableSMul G α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α ν
    s : Set α
    fund_dom_s : MeasureTheory.IsFundamentalDomain G s ν
    ⊢ Eq (MeasureTheory.covolume G α ν) (ν s)
  -/
  dsimp [covolume]
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝³ : Countable G
    inst✝² : MeasurableSpace G
    inst✝¹ : MeasurableSMul G α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α ν
    s : Set α
    fund_dom_s : MeasureTheory.IsFundamentalDomain G s ν
    ⊢ Eq (dite (MeasureTheory.HasFundamentalDomain G α ν) (fun funDom => ν ⋯.choos …
  -/
  simp only [(fund_dom_s.hasFundamentalDomain ν), ↓reduceDIte]
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝³ : Countable G
    inst✝² : MeasurableSpace G
    inst✝¹ : MeasurableSMul G α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α ν
    s : Set α
    fund_dom_s : MeasureTheory.IsFundamentalDomain G s ν
    ⊢ Eq (ν ⋯.choose) (ν s)
  -/
  rw [fund_dom_s.measure_eq]
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝³ : Countable G
    inst✝² : MeasurableSpace G
    inst✝¹ : MeasurableSMul G α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α ν
    s : Set α
    fund_dom_s : MeasureTheory.IsFundamentalDomain G s ν
    ⊢ MeasureTheory.IsFundamentalDomain G ⋯.choose ν
  -/
  exact (fund_dom_s.hasFundamentalDomain ν).ExistsIsFundamentalDomain.choose_spec
  /-
    🎉 no goals
  -/


local notation "α_mod_G" => AddAction.orbitRel G α


/-- A measure `μ` on the `AddQuotient` of `α` mod `G` satisfies
  `AddQuotientMeasureEqMeasurePreimage` if: for any fundamental domain `t`, and any measurable
  subset `U` of the quotient, `μ U = volume ((π ⁻¹' U) ∩ t)`. -/
class AddQuotientMeasureEqMeasurePreimage (ν : Measure α := by volume_tac)
    (μ : Measure (Quotient α_mod_G)) : Prop where
  addProjection_respects_measure' : ∀ (t : Set α) (_ : IsAddFundamentalDomain G t ν),
    μ = (ν.restrict t).map π


/-- Measures `ν` on `α` and `μ` on the `Quotient` of `α` mod `G` satisfy
  `QuotientMeasureEqMeasurePreimage` if: for any fundamental domain `t`, and any measurable subset
  `U` of the quotient, `μ U = ν ((π ⁻¹' U) ∩ t)`. -/
class QuotientMeasureEqMeasurePreimage (ν : Measure α := by volume_tac)
    (μ : Measure (Quotient α_mod_G)) : Prop where
  projection_respects_measure' (t : Set α) : IsFundamentalDomain G t ν → μ = (ν.restrict t).map π


@[to_additive addProjection_respects_measure]
lemma IsFundamentalDomain.projection_respects_measure {ν : Measure α}
    (μ : Measure (Quotient α_mod_G)) [i : QuotientMeasureEqMeasurePreimage ν μ] {t : Set α}
    (fund_dom_t : IsFundamentalDomain G t ν) : μ = (ν.restrict t).map π :=
  i.projection_respects_measure' t fund_dom_t


@[to_additive addProjection_respects_measure_apply]
lemma IsFundamentalDomain.projection_respects_measure_apply {ν : Measure α}
    (μ : Measure (Quotient α_mod_G)) [i : QuotientMeasureEqMeasurePreimage ν μ] {t : Set α}
    (fund_dom_t : IsFundamentalDomain G t ν) {U : Set (Quotient α_mod_G)}
    (meas_U : MeasurableSet U) : μ U = ν (π ⁻¹' U ∩ t) := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝² : Group G
    inst✝¹ : MulAction G α
    inst✝ : MeasurableSpace α
    ν : MeasureTheory.Measure α
    μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
    i : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
    t : Set α
    fund_dom_t : MeasureTheory.IsFundamentalDomain G t ν
    U : Set (Quotient (MulAction.orbitRel G α))
    meas_U : MeasurableSet U
    ⊢ Eq (μ U) (ν (Inter.inter (Set.preimage (Quotient.mk (MulAction.orbitRel G α) …
  -/
  rw [fund_dom_t.projection_respects_measure (μ := μ), measure_map_restrict_apply ν t meas_U]
  /-
    🎉 no goals
  -/


/-- Any two measures satisfying `QuotientMeasureEqMeasurePreimage` are equal. -/
@[to_additive]
lemma QuotientMeasureEqMeasurePreimage.unique
    [hasFun : HasFundamentalDomain G α ν] (μ μ' : Measure (Quotient α_mod_G))
    [QuotientMeasureEqMeasurePreimage ν μ] [QuotientMeasureEqMeasurePreimage ν μ'] :
    μ = μ' := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁴ : Group G
    inst✝³ : MulAction G α
    inst✝² : MeasurableSpace α
    ν : MeasureTheory.Measure α
    hasFun : MeasureTheory.HasFundamentalDomain G α ν
    μ μ' : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
    inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
    inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ'
    ⊢ Eq μ μ'
  -/
  obtain ⟨𝓕, h𝓕⟩ := hasFun.ExistsIsFundamentalDomain
  /-
    case intro
    G : Type u_1
    α : Type u_3
    inst✝⁴ : Group G
    inst✝³ : MulAction G α
    inst✝² : MeasurableSpace α
    ν : MeasureTheory.Measure α
    hasFun : MeasureTheory.HasFundamentalDomain G α ν
    μ μ' : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
    inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
    inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ'
    𝓕 : Set α
    h𝓕 : MeasureTheory.IsFundamentalDomain G 𝓕 ν
    ⊢ Eq μ μ'
  -/
  rw [h𝓕.projection_respects_measure (μ := μ), h𝓕.projection_respects_measure (μ := μ')]
  /-
    🎉 no goals
  -/


/-- The quotient map to `α ⧸ G` is measure-preserving between the restriction of `volume` to a
  fundamental domain in `α` and a related measure satisfying `QuotientMeasureEqMeasurePreimage`. -/
@[to_additive IsAddFundamentalDomain.measurePreserving_add_quotient_mk]
theorem IsFundamentalDomain.measurePreserving_quotient_mk
    {𝓕 : Set α} (h𝓕 : IsFundamentalDomain G 𝓕 ν)
    (μ : Measure (Quotient α_mod_G)) [QuotientMeasureEqMeasurePreimage ν μ] :
    MeasurePreserving π (ν.restrict 𝓕) μ where
  measurable := measurable_quotient_mk' (s := α_mod_G)
  map_eq := by
    /-
      G : Type u_1
      α : Type u_3
      inst✝³ : Group G
      inst✝² : MulAction G α
      inst✝¹ : MeasurableSpace α
      ν : MeasureTheory.Measure α
      𝓕 : Set α
      h𝓕 : MeasureTheory.IsFundamentalDomain G 𝓕 ν
      μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
      inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      ⊢ Eq (MeasureTheory.Measure.map (Quotient.mk (MulAction.orbitRel G α)) (ν.rest …
    -/
    haveI : HasFundamentalDomain G α ν := ⟨𝓕, h𝓕⟩
    /-
      G : Type u_1
      α : Type u_3
      inst✝³ : Group G
      inst✝² : MulAction G α
      inst✝¹ : MeasurableSpace α
      ν : MeasureTheory.Measure α
      𝓕 : Set α
      h𝓕 : MeasureTheory.IsFundamentalDomain G 𝓕 ν
      μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
      inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      this : MeasureTheory.HasFundamentalDomain G α ν
      ⊢ Eq (MeasureTheory.Measure.map (Quotient.mk (MulAction.orbitRel G α)) (ν.rest …
    -/
    rw [h𝓕.projection_respects_measure (μ := μ)]
    /-
      🎉 no goals
    -/


/-- Given a measure upstairs (i.e., on `α`), and a choice `s` of fundamental domain, there's always
an artificial way to generate a measure downstairs such that the pair satisfies the
`QuotientMeasureEqMeasurePreimage` typeclass. -/
@[to_additive]
lemma IsFundamentalDomain.quotientMeasureEqMeasurePreimage_quotientMeasure
    {s : Set α} (fund_dom_s : IsFundamentalDomain G s ν) :
    QuotientMeasureEqMeasurePreimage ν ((ν.restrict s).map π) where
                                                  /-
                                                    G : Type u_1
                                                    α : Type u_3
                                                    inst✝⁶ : Group G
                                                    inst✝⁵ : MulAction G α
                                                    inst✝⁴ : MeasurableSpace α
                                                    ν : MeasureTheory.Measure α
                                                    inst✝³ : MeasureTheory.SMulInvariantMeasure G α ν
                                                    inst✝² : Countable G
                                                    inst✝¹ : MeasurableSpace G
                                                    inst✝ : MeasurableSMul G α
                                                    s : Set α
                                                    fund_dom_s : MeasureTheory.IsFundamentalDomain G s ν
                                                    t : Set α
                                                    fund_dom_t : MeasureTheory.IsFundamentalDomain G t ν
                                                    ⊢ Eq (MeasureTheory.Measure.map (Quotient.mk (MulAction.orbitRel G α)) (ν.rest …
                                                  -/
  projection_respects_measure' t fund_dom_t := by rw [fund_dom_s.quotientMeasure_eq _ fund_dom_t]
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- One can prove `QuotientMeasureEqMeasurePreimage` by checking behavior with respect to a single
fundamental domain. -/
@[to_additive]
lemma IsFundamentalDomain.quotientMeasureEqMeasurePreimage {μ : Measure (Quotient α_mod_G)}
    {s : Set α} (fund_dom_s : IsFundamentalDomain G s ν) (h : μ = (ν.restrict s).map π) :
    QuotientMeasureEqMeasurePreimage ν μ := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝³ : MeasureTheory.SMulInvariantMeasure G α ν
    inst✝² : Countable G
    inst✝¹ : MeasurableSpace G
    inst✝ : MeasurableSMul G α
    μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
    s : Set α
    fund_dom_s : MeasureTheory.IsFundamentalDomain G s ν
    h : Eq μ (MeasureTheory.Measure.map (Quotient.mk (MulAction.orbitRel G α)) (ν. …
    ⊢ MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
  -/
  simpa [h] using fund_dom_s.quotientMeasureEqMeasurePreimage_quotientMeasure
  /-
    🎉 no goals
  -/



/-- If a fundamental domain has volume 0, then `QuotientMeasureEqMeasurePreimage` holds. -/
@[to_additive]
theorem IsFundamentalDomain.quotientMeasureEqMeasurePreimage_of_zero
    {s : Set α} (fund_dom_s : IsFundamentalDomain G s ν)
    (vol_s : ν s = 0) :
    QuotientMeasureEqMeasurePreimage ν (0 : Measure (Quotient α_mod_G)) := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝³ : MeasureTheory.SMulInvariantMeasure G α ν
    inst✝² : Countable G
    inst✝¹ : MeasurableSpace G
    inst✝ : MeasurableSMul G α
    s : Set α
    fund_dom_s : MeasureTheory.IsFundamentalDomain G s ν
    vol_s : Eq (ν s) 0
    ⊢ MeasureTheory.QuotientMeasureEqMeasurePreimage ν 0
  -/
  apply fund_dom_s.quotientMeasureEqMeasurePreimage
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝³ : MeasureTheory.SMulInvariantMeasure G α ν
    inst✝² : Countable G
    inst✝¹ : MeasurableSpace G
    inst✝ : MeasurableSMul G α
    s : Set α
    fund_dom_s : MeasureTheory.IsFundamentalDomain G s ν
    vol_s : Eq (ν s) 0
    ⊢ Eq 0 (MeasureTheory.Measure.map (Quotient.mk (MulAction.orbitRel G α)) (ν.re …
  -/
  ext U meas_U
  /-
    case h
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝³ : MeasureTheory.SMulInvariantMeasure G α ν
    inst✝² : Countable G
    inst✝¹ : MeasurableSpace G
    inst✝ : MeasurableSMul G α
    s : Set α
    fund_dom_s : MeasureTheory.IsFundamentalDomain G s ν
    vol_s : Eq (ν s) 0
    U : Set (Quotient (MulAction.orbitRel G α))
    meas_U : MeasurableSet U
    ⊢ Eq (0 U) ((MeasureTheory.Measure.map (Quotient.mk (MulAction.orbitRel G α))  …
  -/
  simp only [Measure.coe_zero, Pi.zero_apply]
  /-
    case h
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝³ : MeasureTheory.SMulInvariantMeasure G α ν
    inst✝² : Countable G
    inst✝¹ : MeasurableSpace G
    inst✝ : MeasurableSMul G α
    s : Set α
    fund_dom_s : MeasureTheory.IsFundamentalDomain G s ν
    vol_s : Eq (ν s) 0
    U : Set (Quotient (MulAction.orbitRel G α))
    meas_U : MeasurableSet U
    ⊢ Eq 0 ((MeasureTheory.Measure.map (Quotient.mk (MulAction.orbitRel G α)) (ν.r …
  -/
  convert (measure_inter_null_of_null_right (h := vol_s) (Quotient.mk α_mod_G ⁻¹' U)).symm
  /-
    case h.e'_3
    G : Type u_1
    α : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : MulAction G α
    inst✝⁴ : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝³ : MeasureTheory.SMulInvariantMeasure G α ν
    inst✝² : Countable G
    inst✝¹ : MeasurableSpace G
    inst✝ : MeasurableSMul G α
    s : Set α
    fund_dom_s : MeasureTheory.IsFundamentalDomain G s ν
    vol_s : Eq (ν s) 0
    U : Set (Quotient (MulAction.orbitRel G α))
    meas_U : MeasurableSet U
    ⊢ Eq ((MeasureTheory.Measure.map (Quotient.mk (MulAction.orbitRel G α)) (ν.res …
  -/
  rw [measure_map_restrict_apply (meas_U := meas_U)]
  /-
    🎉 no goals
  -/


/-- If a measure `μ` on a quotient satisfies `QuotientMeasureEqMeasurePreimage` with respect to a
sigma-finite measure `ν`, then it is itself `SigmaFinite`. -/
@[to_additive]
lemma QuotientMeasureEqMeasurePreimage.sigmaFiniteQuotient
    [i : SigmaFinite ν] [i' : HasFundamentalDomain G α ν]
    (μ : Measure (Quotient α_mod_G)) [QuotientMeasureEqMeasurePreimage ν μ] :
    SigmaFinite μ := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁷ : Group G
    inst✝⁶ : MulAction G α
    inst✝⁵ : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝⁴ : MeasureTheory.SMulInvariantMeasure G α ν
    inst✝³ : Countable G
    inst✝² : MeasurableSpace G
    inst✝¹ : MeasurableSMul G α
    i : MeasureTheory.SigmaFinite ν
    i' : MeasureTheory.HasFundamentalDomain G α ν
    μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
    inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
    ⊢ MeasureTheory.SigmaFinite μ
  -/
  rw [sigmaFinite_iff]
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁷ : Group G
    inst✝⁶ : MulAction G α
    inst✝⁵ : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝⁴ : MeasureTheory.SMulInvariantMeasure G α ν
    inst✝³ : Countable G
    inst✝² : MeasurableSpace G
    inst✝¹ : MeasurableSMul G α
    i : MeasureTheory.SigmaFinite ν
    i' : MeasureTheory.HasFundamentalDomain G α ν
    μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
    inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
    ⊢ Nonempty (μ.FiniteSpanningSetsIn Set.univ)
  -/
  obtain ⟨A, hA_meas, hA, hA'⟩ := Measure.toFiniteSpanningSetsIn (h := i)
  /-
    case mk
    G : Type u_1
    α : Type u_3
    inst✝⁷ : Group G
    inst✝⁶ : MulAction G α
    inst✝⁵ : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝⁴ : MeasureTheory.SMulInvariantMeasure G α ν
    inst✝³ : Countable G
    inst✝² : MeasurableSpace G
    inst✝¹ : MeasurableSMul G α
    i : MeasureTheory.SigmaFinite ν
    i' : MeasureTheory.HasFundamentalDomain G α ν
    μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
    inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
    A : Nat → Set α
    hA_meas : ∀ (i : Nat), Membership.mem (setOf fun s => MeasurableSet s) (A i)
    hA : ∀ (i : Nat), LT.lt (ν (A i)) Top.top
    hA' : Eq (Set.iUnion fun i => A i) Set.univ
    ⊢ Nonempty (μ.FiniteSpanningSetsIn Set.univ)
  -/
  simp only [mem_setOf_eq] at hA_meas
  /-
    case mk
    G : Type u_1
    α : Type u_3
    inst✝⁷ : Group G
    inst✝⁶ : MulAction G α
    inst✝⁵ : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝⁴ : MeasureTheory.SMulInvariantMeasure G α ν
    inst✝³ : Countable G
    inst✝² : MeasurableSpace G
    inst✝¹ : MeasurableSMul G α
    i : MeasureTheory.SigmaFinite ν
    i' : MeasureTheory.HasFundamentalDomain G α ν
    μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
    inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
    A : Nat → Set α
    hA_meas : ∀ (i : Nat), MeasurableSet (A i)
    hA : ∀ (i : Nat), LT.lt (ν (A i)) Top.top
    hA' : Eq (Set.iUnion fun i => A i) Set.univ
    ⊢ Nonempty (μ.FiniteSpanningSetsIn Set.univ)
  -/
  refine ⟨⟨fun n ↦ π '' (A n), by simp, fun n ↦ ?_, ?_⟩⟩
    /-
      case mk.refine_1
      G : Type u_1
      α : Type u_3
      inst✝⁷ : Group G
      inst✝⁶ : MulAction G α
      inst✝⁵ : MeasurableSpace α
      ν : MeasureTheory.Measure α
      inst✝⁴ : MeasureTheory.SMulInvariantMeasure G α ν
      inst✝³ : Countable G
      inst✝² : MeasurableSpace G
      inst✝¹ : MeasurableSMul G α
      i : MeasureTheory.SigmaFinite ν
      i' : MeasureTheory.HasFundamentalDomain G α ν
      μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
      inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      A : Nat → Set α
      hA_meas : ∀ (i : Nat), MeasurableSet (A i)
      hA : ∀ (i : Nat), LT.lt (ν (A i)) Top.top
      hA' : Eq (Set.iUnion fun i => A i) Set.univ
      n : Nat
      ⊢ LT.lt (μ ((fun n => Set.image (Quotient.mk (MulAction.orbitRel G α)) (A n))  …
    -/
  · obtain ⟨s, fund_dom_s⟩ := i'
    /-
      case mk.refine_1.mk.intro
      G : Type u_1
      α : Type u_3
      inst✝⁷ : Group G
      inst✝⁶ : MulAction G α
      inst✝⁵ : MeasurableSpace α
      ν : MeasureTheory.Measure α
      inst✝⁴ : MeasureTheory.SMulInvariantMeasure G α ν
      inst✝³ : Countable G
      inst✝² : MeasurableSpace G
      inst✝¹ : MeasurableSMul G α
      i : MeasureTheory.SigmaFinite ν
      μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
      inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      A : Nat → Set α
      hA_meas : ∀ (i : Nat), MeasurableSet (A i)
      hA : ∀ (i : Nat), LT.lt (ν (A i)) Top.top
      hA' : Eq (Set.iUnion fun i => A i) Set.univ
      n : Nat
      s : Set α
      fund_dom_s : MeasureTheory.IsFundamentalDomain G s ν
      ⊢ LT.lt (μ ((fun n => Set.image (Quotient.mk (MulAction.orbitRel G α)) (A n))  …
    -/
    have : π ⁻¹' (π '' (A n)) = _ := MulAction.quotient_preimage_image_eq_union_mul (A n) (G := G)
    have measπAn : MeasurableSet (π '' A n) := by
      let _ : Setoid α := α_mod_G
      rw [measurableSet_quotient, Quotient.mk''_eq_mk, this]
      apply MeasurableSet.iUnion
      exact fun g ↦ MeasurableSet.const_smul (hA_meas n) g
    /-
      case mk.refine_1.mk.intro
      G : Type u_1
      α : Type u_3
      inst✝⁷ : Group G
      inst✝⁶ : MulAction G α
      inst✝⁵ : MeasurableSpace α
      ν : MeasureTheory.Measure α
      inst✝⁴ : MeasureTheory.SMulInvariantMeasure G α ν
      inst✝³ : Countable G
      inst✝² : MeasurableSpace G
      inst✝¹ : MeasurableSMul G α
      i : MeasureTheory.SigmaFinite ν
      μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
      inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      A : Nat → Set α
      hA_meas : ∀ (i : Nat), MeasurableSet (A i)
      hA : ∀ (i : Nat), LT.lt (ν (A i)) Top.top
      hA' : Eq (Set.iUnion fun i => A i) Set.univ
      n : Nat
      s : Set α
      fund_dom_s : MeasureTheory.IsFundamentalDomain G s ν
      this : Eq (Set.preimage (Quotient.mk (MulAction.orbitRel G α)) (Set.image (Quo …
      measπAn : MeasurableSet (Set.image (Quotient.mk (MulAction.orbitRel G α)) (A n))
      ⊢ LT.lt (μ ((fun n => Set.image (Quotient.mk (MulAction.orbitRel G α)) (A n))  …
    -/
    rw [fund_dom_s.projection_respects_measure_apply (μ := μ) measπAn, this, iUnion_inter]
    /-
      case mk.refine_1.mk.intro
      G : Type u_1
      α : Type u_3
      inst✝⁷ : Group G
      inst✝⁶ : MulAction G α
      inst✝⁵ : MeasurableSpace α
      ν : MeasureTheory.Measure α
      inst✝⁴ : MeasureTheory.SMulInvariantMeasure G α ν
      inst✝³ : Countable G
      inst✝² : MeasurableSpace G
      inst✝¹ : MeasurableSMul G α
      i : MeasureTheory.SigmaFinite ν
      μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
      inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      A : Nat → Set α
      hA_meas : ∀ (i : Nat), MeasurableSet (A i)
      hA : ∀ (i : Nat), LT.lt (ν (A i)) Top.top
      hA' : Eq (Set.iUnion fun i => A i) Set.univ
      n : Nat
      s : Set α
      fund_dom_s : MeasureTheory.IsFundamentalDomain G s ν
      this : Eq (Set.preimage (Quotient.mk (MulAction.orbitRel G α)) (Set.image (Quo …
      measπAn : MeasurableSet (Set.image (Quotient.mk (MulAction.orbitRel G α)) (A n))
      ⊢ LT.lt (ν (Set.iUnion fun i => Inter.inter (Set.image (fun x => HSMul.hSMul i …
    -/
    refine lt_of_le_of_lt ?_ (hA n)
    /-
      case mk.refine_1.mk.intro
      G : Type u_1
      α : Type u_3
      inst✝⁷ : Group G
      inst✝⁶ : MulAction G α
      inst✝⁵ : MeasurableSpace α
      ν : MeasureTheory.Measure α
      inst✝⁴ : MeasureTheory.SMulInvariantMeasure G α ν
      inst✝³ : Countable G
      inst✝² : MeasurableSpace G
      inst✝¹ : MeasurableSMul G α
      i : MeasureTheory.SigmaFinite ν
      μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
      inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      A : Nat → Set α
      hA_meas : ∀ (i : Nat), MeasurableSet (A i)
      hA : ∀ (i : Nat), LT.lt (ν (A i)) Top.top
      hA' : Eq (Set.iUnion fun i => A i) Set.univ
      n : Nat
      s : Set α
      fund_dom_s : MeasureTheory.IsFundamentalDomain G s ν
      this : Eq (Set.preimage (Quotient.mk (MulAction.orbitRel G α)) (Set.image (Quo …
      measπAn : MeasurableSet (Set.image (Quotient.mk (MulAction.orbitRel G α)) (A n))
      ⊢ LE.le (ν (Set.iUnion fun i => Inter.inter (Set.image (fun x => HSMul.hSMul i …
    -/
    rw [fund_dom_s.measure_eq_tsum (A n)]
    /-
      case mk.refine_1.mk.intro
      G : Type u_1
      α : Type u_3
      inst✝⁷ : Group G
      inst✝⁶ : MulAction G α
      inst✝⁵ : MeasurableSpace α
      ν : MeasureTheory.Measure α
      inst✝⁴ : MeasureTheory.SMulInvariantMeasure G α ν
      inst✝³ : Countable G
      inst✝² : MeasurableSpace G
      inst✝¹ : MeasurableSMul G α
      i : MeasureTheory.SigmaFinite ν
      μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
      inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      A : Nat → Set α
      hA_meas : ∀ (i : Nat), MeasurableSet (A i)
      hA : ∀ (i : Nat), LT.lt (ν (A i)) Top.top
      hA' : Eq (Set.iUnion fun i => A i) Set.univ
      n : Nat
      s : Set α
      fund_dom_s : MeasureTheory.IsFundamentalDomain G s ν
      this : Eq (Set.preimage (Quotient.mk (MulAction.orbitRel G α)) (Set.image (Quo …
      measπAn : MeasurableSet (Set.image (Quotient.mk (MulAction.orbitRel G α)) (A n))
      ⊢ LE.le (ν (Set.iUnion fun i => Inter.inter (Set.image (fun x => HSMul.hSMul i …
    -/
    exact measure_iUnion_le _
    /-
      🎉 no goals
    -/
    /-
      case mk.refine_2
      G : Type u_1
      α : Type u_3
      inst✝⁷ : Group G
      inst✝⁶ : MulAction G α
      inst✝⁵ : MeasurableSpace α
      ν : MeasureTheory.Measure α
      inst✝⁴ : MeasureTheory.SMulInvariantMeasure G α ν
      inst✝³ : Countable G
      inst✝² : MeasurableSpace G
      inst✝¹ : MeasurableSMul G α
      i : MeasureTheory.SigmaFinite ν
      i' : MeasureTheory.HasFundamentalDomain G α ν
      μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
      inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      A : Nat → Set α
      hA_meas : ∀ (i : Nat), MeasurableSet (A i)
      hA : ∀ (i : Nat), LT.lt (ν (A i)) Top.top
      hA' : Eq (Set.iUnion fun i => A i) Set.univ
      ⊢ Eq (Set.iUnion fun i => (fun n => Set.image (Quotient.mk (MulAction.orbitRel …
    -/
  · rw [← image_iUnion, hA']
    /-
      case mk.refine_2
      G : Type u_1
      α : Type u_3
      inst✝⁷ : Group G
      inst✝⁶ : MulAction G α
      inst✝⁵ : MeasurableSpace α
      ν : MeasureTheory.Measure α
      inst✝⁴ : MeasureTheory.SMulInvariantMeasure G α ν
      inst✝³ : Countable G
      inst✝² : MeasurableSpace G
      inst✝¹ : MeasurableSMul G α
      i : MeasureTheory.SigmaFinite ν
      i' : MeasureTheory.HasFundamentalDomain G α ν
      μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
      inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      A : Nat → Set α
      hA_meas : ∀ (i : Nat), MeasurableSet (A i)
      hA : ∀ (i : Nat), LT.lt (ν (A i)) Top.top
      hA' : Eq (Set.iUnion fun i => A i) Set.univ
      ⊢ Eq (Set.image (Quotient.mk (MulAction.orbitRel G α)) Set.univ) Set.univ
    -/
    refine image_univ_of_surjective (by convert Quotient.mk'_surjective)
    /-
      🎉 no goals
    -/


/-- A measure `μ` on `α ⧸ G` satisfying `QuotientMeasureEqMeasurePreimage` and having finite
covolume is a finite measure. -/
@[to_additive]
theorem QuotientMeasureEqMeasurePreimage.isFiniteMeasure_quotient
    (μ : Measure (Quotient α_mod_G)) [QuotientMeasureEqMeasurePreimage ν μ]
    [hasFun : HasFundamentalDomain G α ν] (h : covolume G α ν ≠ ∞) :
    IsFiniteMeasure μ := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁷ : Group G
    inst✝⁶ : MulAction G α
    inst✝⁵ : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝⁴ : MeasureTheory.SMulInvariantMeasure G α ν
    inst✝³ : Countable G
    inst✝² : MeasurableSpace G
    inst✝¹ : MeasurableSMul G α
    μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
    inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
    hasFun : MeasureTheory.HasFundamentalDomain G α ν
    h : Ne (MeasureTheory.covolume G α ν) Top.top
    ⊢ MeasureTheory.IsFiniteMeasure μ
  -/
  obtain ⟨𝓕, h𝓕⟩ := hasFun.ExistsIsFundamentalDomain
  /-
    case intro
    G : Type u_1
    α : Type u_3
    inst✝⁷ : Group G
    inst✝⁶ : MulAction G α
    inst✝⁵ : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝⁴ : MeasureTheory.SMulInvariantMeasure G α ν
    inst✝³ : Countable G
    inst✝² : MeasurableSpace G
    inst✝¹ : MeasurableSMul G α
    μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
    inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
    hasFun : MeasureTheory.HasFundamentalDomain G α ν
    h : Ne (MeasureTheory.covolume G α ν) Top.top
    𝓕 : Set α
    h𝓕 : MeasureTheory.IsFundamentalDomain G 𝓕 ν
    ⊢ MeasureTheory.IsFiniteMeasure μ
  -/
  rw [h𝓕.projection_respects_measure (μ := μ)]
  have : Fact (ν 𝓕 < ∞) := by
    apply Fact.mk
    convert Ne.lt_top h
    exact (h𝓕.covolume_eq_volume ν).symm
  /-
    case intro
    G : Type u_1
    α : Type u_3
    inst✝⁷ : Group G
    inst✝⁶ : MulAction G α
    inst✝⁵ : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝⁴ : MeasureTheory.SMulInvariantMeasure G α ν
    inst✝³ : Countable G
    inst✝² : MeasurableSpace G
    inst✝¹ : MeasurableSMul G α
    μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
    inst✝ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
    hasFun : MeasureTheory.HasFundamentalDomain G α ν
    h : Ne (MeasureTheory.covolume G α ν) Top.top
    𝓕 : Set α
    h𝓕 : MeasureTheory.IsFundamentalDomain G 𝓕 ν
    this : Fact (LT.lt (ν 𝓕) Top.top)
    ⊢ MeasureTheory.IsFiniteMeasure (MeasureTheory.Measure.map (Quotient.mk (MulAc …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- A finite measure `μ` on `α ⧸ G` satisfying `QuotientMeasureEqMeasurePreimage` has finite
covolume. -/
@[to_additive]
theorem QuotientMeasureEqMeasurePreimage.covolume_ne_top
    (μ : Measure (Quotient α_mod_G)) [QuotientMeasureEqMeasurePreimage ν μ] [IsFiniteMeasure μ] :
    covolume G α ν < ∞ := by
  /-
    G : Type u_1
    α : Type u_3
    inst✝⁸ : Group G
    inst✝⁷ : MulAction G α
    inst✝⁶ : MeasurableSpace α
    ν : MeasureTheory.Measure α
    inst✝⁵ : MeasureTheory.SMulInvariantMeasure G α ν
    inst✝⁴ : Countable G
    inst✝³ : MeasurableSpace G
    inst✝² : MeasurableSMul G α
    μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
    inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    ⊢ LT.lt (MeasureTheory.covolume G α ν) Top.top
  -/
  by_cases hasFun : HasFundamentalDomain G α ν
    /-
      case pos
      G : Type u_1
      α : Type u_3
      inst✝⁸ : Group G
      inst✝⁷ : MulAction G α
      inst✝⁶ : MeasurableSpace α
      ν : MeasureTheory.Measure α
      inst✝⁵ : MeasureTheory.SMulInvariantMeasure G α ν
      inst✝⁴ : Countable G
      inst✝³ : MeasurableSpace G
      inst✝² : MeasurableSMul G α
      μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
      inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hasFun : MeasureTheory.HasFundamentalDomain G α ν
      ⊢ LT.lt (MeasureTheory.covolume G α ν) Top.top
    -/
  · obtain ⟨𝓕, h𝓕⟩ := hasFun.ExistsIsFundamentalDomain
    /-
      case pos.intro
      G : Type u_1
      α : Type u_3
      inst✝⁸ : Group G
      inst✝⁷ : MulAction G α
      inst✝⁶ : MeasurableSpace α
      ν : MeasureTheory.Measure α
      inst✝⁵ : MeasureTheory.SMulInvariantMeasure G α ν
      inst✝⁴ : Countable G
      inst✝³ : MeasurableSpace G
      inst✝² : MeasurableSMul G α
      μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
      inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hasFun : MeasureTheory.HasFundamentalDomain G α ν
      𝓕 : Set α
      h𝓕 : MeasureTheory.IsFundamentalDomain G 𝓕 ν
      ⊢ LT.lt (MeasureTheory.covolume G α ν) Top.top
    -/
    have H : μ univ < ∞ := IsFiniteMeasure.measure_univ_lt_top
    /-
      case pos.intro
      G : Type u_1
      α : Type u_3
      inst✝⁸ : Group G
      inst✝⁷ : MulAction G α
      inst✝⁶ : MeasurableSpace α
      ν : MeasureTheory.Measure α
      inst✝⁵ : MeasureTheory.SMulInvariantMeasure G α ν
      inst✝⁴ : Countable G
      inst✝³ : MeasurableSpace G
      inst✝² : MeasurableSMul G α
      μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
      inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hasFun : MeasureTheory.HasFundamentalDomain G α ν
      𝓕 : Set α
      h𝓕 : MeasureTheory.IsFundamentalDomain G 𝓕 ν
      H : LT.lt (μ Set.univ) Top.top
      ⊢ LT.lt (MeasureTheory.covolume G α ν) Top.top
    -/
    rw [h𝓕.projection_respects_measure_apply (μ := μ) MeasurableSet.univ] at H
    /-
      case pos.intro
      G : Type u_1
      α : Type u_3
      inst✝⁸ : Group G
      inst✝⁷ : MulAction G α
      inst✝⁶ : MeasurableSpace α
      ν : MeasureTheory.Measure α
      inst✝⁵ : MeasureTheory.SMulInvariantMeasure G α ν
      inst✝⁴ : Countable G
      inst✝³ : MeasurableSpace G
      inst✝² : MeasurableSMul G α
      μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
      inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hasFun : MeasureTheory.HasFundamentalDomain G α ν
      𝓕 : Set α
      h𝓕 : MeasureTheory.IsFundamentalDomain G 𝓕 ν
      H : LT.lt (ν (Inter.inter (Set.preimage (Quotient.mk (MulAction.orbitRel G α)) …
      ⊢ LT.lt (MeasureTheory.covolume G α ν) Top.top
    -/
    simpa [h𝓕.covolume_eq_volume ν] using H
    /-
      🎉 no goals
    -/
    /-
      case neg
      G : Type u_1
      α : Type u_3
      inst✝⁸ : Group G
      inst✝⁷ : MulAction G α
      inst✝⁶ : MeasurableSpace α
      ν : MeasureTheory.Measure α
      inst✝⁵ : MeasureTheory.SMulInvariantMeasure G α ν
      inst✝⁴ : Countable G
      inst✝³ : MeasurableSpace G
      inst✝² : MeasurableSMul G α
      μ : MeasureTheory.Measure (Quotient (MulAction.orbitRel G α))
      inst✝¹ : MeasureTheory.QuotientMeasureEqMeasurePreimage ν μ
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hasFun : Not (MeasureTheory.HasFundamentalDomain G α ν)
      ⊢ LT.lt (MeasureTheory.covolume G α ν) Top.top
    -/
  · simp [covolume, hasFun]
    /-
      🎉 no goals
    -/


/-- If a measure `μ` on a quotient satisfies `QuotientVolumeEqVolumePreimage` with respect to a
sigma-finite measure, then it is itself `SigmaFinite`. -/
@[to_additive MeasureTheory.instSigmaFiniteAddQuotientOrbitRelInstMeasurableSpaceToMeasurableSpace]
                                             /-
                                               G : Type u_1
                                               H : Type u_2
                                               α : Type u_3
                                               β : Type u_4
                                               E : Type u_5
                                               inst✝⁷ : Group G
                                               inst✝⁶ : MulAction G α
                                               inst✝⁵ : MeasureTheory.MeasureSpace α
                                               inst✝⁴ : Countable G
                                               inst✝³ : MeasurableSpace G
                                               inst✝² : MeasureTheory.SMulInvariantMeasure G α MeasureTheory.MeasureSpace.vol …
                                               inst✝¹ : MeasurableSMul G α
                                               inst✝ : MeasureTheory.SigmaFinite MeasureTheory.MeasureSpace.volume
                                               ⊢ MeasureTheory.Measure α
                                             -/
instance [SigmaFinite (volume : Measure α)] [HasFundamentalDomain G α]
                                             /-
                                               🎉 no goals
                                             -/
    (μ : Measure (Quotient α_mod_G)) [QuotientMeasureEqMeasurePreimage volume μ] :
    SigmaFinite μ :=
  QuotientMeasureEqMeasurePreimage.sigmaFiniteQuotient (ν := (volume : Measure α)) (μ := μ)



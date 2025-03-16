/--
The extended diameter is the greatest distance between any two vertices, with the value `⊤` in
case the distances are not bounded above, or the graph is not connected.
-/
noncomputable def ediam (G : SimpleGraph α) : ℕ∞ :=
  ⨆ u, ⨆ v, G.edist u v


lemma ediam_def : G.ediam = ⨆ p : α × α, G.edist p.1 p.2 := by
  /-
    α : Type u_1
    G : SimpleGraph α
    ⊢ Eq G.ediam (iSup fun p => G.edist p.1 p.2)
  -/
  rw [ediam, iSup_prod]
  /-
    🎉 no goals
  -/


lemma edist_le_ediam {u v : α} : G.edist u v ≤ G.ediam :=
  le_iSup₂ (f := G.edist) u v


lemma ediam_le_of_edist_le {k : ℕ∞} (h : ∀ u v, G.edist u v ≤ k ) : G.ediam ≤ k :=
  iSup₂_le h


lemma ediam_le_iff {k : ℕ∞} : G.ediam ≤ k ↔ ∀ u v, G.edist u v ≤ k :=
  iSup₂_le_iff


lemma ediam_eq_top : G.ediam = ⊤ ↔ ∀ b < ⊤, ∃ u v, b < G.edist u v := by
  /-
    α : Type u_1
    G : SimpleGraph α
    ⊢ Iff (Eq G.ediam Top.top) (∀ (b : ENat), LT.lt b Top.top → Exists fun u => Ex …
  -/
  simp only [ediam, iSup_eq_top, lt_iSup_iff]
  /-
    🎉 no goals
  -/


lemma ediam_eq_zero_of_subsingleton [Subsingleton α] : G.ediam = 0 := by
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Subsingleton α
    ⊢ Eq G.ediam 0
  -/
  rw [ediam_def, ENat.iSup_eq_zero]
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Subsingleton α
    ⊢ ∀ (i : Prod α α), Eq (G.edist i.1 i.2) 0
  -/
  simpa [edist_eq_zero_iff, Prod.forall] using subsingleton_iff.mp ‹_›
  /-
    🎉 no goals
  -/


lemma nontrivial_of_ediam_ne_zero (h : G.ediam ≠ 0) : Nontrivial α := by
  /-
    α : Type u_1
    G : SimpleGraph α
    h : Ne G.ediam 0
    ⊢ Nontrivial α
  -/
  contrapose! h
  /-
    α : Type u_1
    G : SimpleGraph α
    h : Not (Nontrivial α)
    ⊢ Eq G.ediam 0
  -/
  rw [not_nontrivial_iff_subsingleton] at h
  /-
    α : Type u_1
    G : SimpleGraph α
    h : Subsingleton α
    ⊢ Eq G.ediam 0
  -/
  exact ediam_eq_zero_of_subsingleton
  /-
    🎉 no goals
  -/


lemma ediam_ne_zero [Nontrivial α] : G.ediam ≠ 0 := by
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Nontrivial α
    ⊢ Ne G.ediam 0
  -/
  obtain ⟨u, v, huv⟩ := exists_pair_ne ‹_›
  /-
    case intro.intro
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Nontrivial α
    u v : α
    huv : Ne u v
    ⊢ Ne G.ediam 0
  -/
  contrapose! huv
  /-
    case intro.intro
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Nontrivial α
    u v : α
    huv : Eq G.ediam 0
    ⊢ Eq u v
  -/
  simp only [ediam, nonpos_iff_eq_zero, ENat.iSup_eq_zero, edist_eq_zero_iff] at huv
  /-
    case intro.intro
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Nontrivial α
    u v : α
    huv : ∀ (i i_1 : α), Eq i i_1
    ⊢ Eq u v
  -/
  exact huv u v
  /-
    🎉 no goals
  -/


lemma subsingleton_of_ediam_eq_zero (h : G.ediam = 0) : Subsingleton α := by
  /-
    α : Type u_1
    G : SimpleGraph α
    h : Eq G.ediam 0
    ⊢ Subsingleton α
  -/
  contrapose! h
  /-
    α : Type u_1
    G : SimpleGraph α
    h : Not (Subsingleton α)
    ⊢ Ne G.ediam 0
  -/
  apply not_subsingleton_iff_nontrivial.mp at h
  /-
    α : Type u_1
    G : SimpleGraph α
    h : Nontrivial α
    ⊢ Ne G.ediam 0
  -/
  exact ediam_ne_zero
  /-
    🎉 no goals
  -/


lemma ediam_ne_zero_iff_nontrivial :
    G.ediam ≠ 0 ↔ Nontrivial α :=
  ⟨nontrivial_of_ediam_ne_zero, fun _ ↦ ediam_ne_zero⟩


@[simp]
lemma ediam_eq_zero_iff_subsingleton :
    G.ediam = 0 ↔ Subsingleton α :=
  ⟨subsingleton_of_ediam_eq_zero, fun _ ↦ ediam_eq_zero_of_subsingleton⟩


lemma ediam_eq_top_of_not_connected [Nonempty α] (h : ¬G.Connected) : G.ediam = ⊤ := by
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Nonempty α
    h : Not G.Connected
    ⊢ Eq G.ediam Top.top
  -/
  rw [connected_iff_exists_forall_reachable] at h
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Nonempty α
    h : Not (Exists fun v => ∀ (w : α), G.Reachable v w)
    ⊢ Eq G.ediam Top.top
  -/
  push_neg at h
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Nonempty α
    h : ∀ (v : α), Exists fun w => Not (G.Reachable v w)
    ⊢ Eq G.ediam Top.top
  -/
  obtain ⟨_, hw⟩ := h Classical.ofNonempty
  /-
    case intro
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Nonempty α
    h : ∀ (v : α), Exists fun w => Not (G.Reachable v w)
    w✝ : α
    hw : Not (G.Reachable Classical.ofNonempty w✝)
    ⊢ Eq G.ediam Top.top
  -/
  rw [eq_top_iff, ← edist_eq_top_of_not_reachable hw]
  /-
    case intro
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Nonempty α
    h : ∀ (v : α), Exists fun w => Not (G.Reachable v w)
    w✝ : α
    hw : Not (G.Reachable Classical.ofNonempty w✝)
    ⊢ LE.le (G.edist Classical.ofNonempty w✝) G.ediam
  -/
  exact edist_le_ediam
  /-
    🎉 no goals
  -/


lemma ediam_eq_top_of_not_preconnected (h : ¬G.Preconnected) : G.ediam = ⊤ := by
  /-
    α : Type u_1
    G : SimpleGraph α
    h : Not G.Preconnected
    ⊢ Eq G.ediam Top.top
  -/
  cases isEmpty_or_nonempty α
    /-
      case inl
      α : Type u_1
      G : SimpleGraph α
      h : Not G.Preconnected
      h✝ : IsEmpty α
      ⊢ Eq G.ediam Top.top
    -/
  · exfalso
    /-
      case inl
      α : Type u_1
      G : SimpleGraph α
      h : Not G.Preconnected
      h✝ : IsEmpty α
      ⊢ False
    -/
    exact h <| IsEmpty.forall_iff.mpr trivial
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      G : SimpleGraph α
      h : Not G.Preconnected
      h✝ : Nonempty α
      ⊢ Eq G.ediam Top.top
    -/
  · apply ediam_eq_top_of_not_connected
    /-
      case inr.h
      α : Type u_1
      G : SimpleGraph α
      h : Not G.Preconnected
      h✝ : Nonempty α
      ⊢ Not G.Connected
    -/
    rw [connected_iff]
    /-
      case inr.h
      α : Type u_1
      G : SimpleGraph α
      h : Not G.Preconnected
      h✝ : Nonempty α
      ⊢ Not (And G.Preconnected (Nonempty α))
    -/
    tauto
    /-
      🎉 no goals
    -/


lemma exists_edist_eq_ediam_of_ne_top [Nonempty α] (h : G.ediam ≠ ⊤) :
    ∃ u v, G.edist u v = G.ediam :=
  ENat.exists_eq_iSup₂_of_lt_top h.lt_top

-- Note: Neither `Finite α` nor `G.ediam ≠ ⊤` implies the other.

lemma exists_edist_eq_ediam_of_finite [Nonempty α] [Finite α] :
    ∃ u v, G.edist u v = G.ediam :=
  Prod.exists'.mp <| ediam_def ▸ exists_eq_ciSup_of_finite


@[gcongr]
lemma ediam_anti (h : G ≤ G') : G'.ediam ≤ G.ediam :=
  iSup₂_mono fun _ _ ↦ edist_anti h


@[simp]
lemma ediam_bot [Nontrivial α] : (⊥ : SimpleGraph α).ediam = ⊤ :=
  ediam_eq_top_of_not_connected bot_not_connected


@[simp]
lemma ediam_top [Nontrivial α] : (⊤ : SimpleGraph α).ediam = 1 := by
  /-
    α : Type u_1
    inst✝ : Nontrivial α
    ⊢ Eq Top.top.ediam 1
  -/
  apply le_antisymm ?_ <| Order.one_le_iff_pos.mpr <| pos_iff_ne_zero.mpr ediam_ne_zero
  /-
    α : Type u_1
    inst✝ : Nontrivial α
    ⊢ LE.le Top.top.ediam 1
  -/
  apply ediam_def ▸ iSup_le_iff.mpr
  /-
    α : Type u_1
    inst✝ : Nontrivial α
    ⊢ ∀ (i : Prod α α), LE.le (Top.top.edist i.1 i.2) 1
  -/
  intro p
  /-
    α : Type u_1
    inst✝ : Nontrivial α
    p : Prod α α
    ⊢ LE.le (Top.top.edist p.1 p.2) 1
  -/
  by_cases h : (⊤ : SimpleGraph α).Adj p.1 p.2
    /-
      case pos
      α : Type u_1
      inst✝ : Nontrivial α
      p : Prod α α
      h : Top.top.Adj p.1 p.2
      ⊢ LE.le (Top.top.edist p.1 p.2) 1
    -/
  · apply le_of_eq <| edist_eq_one_iff_adj.mpr h
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : Nontrivial α
      p : Prod α α
      h : Not (Top.top.Adj p.1 p.2)
      ⊢ LE.le (Top.top.edist p.1 p.2) 1
    -/
  · simp_all
    /-
      🎉 no goals
    -/


@[simp]
lemma ediam_eq_one [Nontrivial α] : G.ediam = 1 ↔ G = ⊤ := by
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Nontrivial α
    ⊢ Iff (Eq G.ediam 1) (Eq G Top.top)
  -/
  refine ⟨fun h₁ ↦ ?_, fun h ↦ h ▸ ediam_top⟩
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Nontrivial α
    h₁ : Eq G.ediam 1
    ⊢ Eq G Top.top
  -/
  ext u v
  /-
    case Adj.h.h.a
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Nontrivial α
    h₁ : Eq G.ediam 1
    u v : α
    ⊢ Iff (G.Adj u v) (Top.top.Adj u v)
  -/
  refine ⟨fun h ↦ h.ne, fun h₂ ↦ ?_⟩
  /-
    case Adj.h.h.a
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Nontrivial α
    h₁ : Eq G.ediam 1
    u v : α
    h₂ : Top.top.Adj u v
    ⊢ G.Adj u v
  -/
  apply G.edist_pos_of_ne at h₂
  /-
    case Adj.h.h.a
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Nontrivial α
    h₁ : Eq G.ediam 1
    u v : α
    h₂ : LT.lt 0 (G.edist u v)
    ⊢ G.Adj u v
  -/
  apply le_of_eq at h₁
  /-
    case Adj.h.h.a
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Nontrivial α
    u v : α
    h₂ : LT.lt 0 (G.edist u v)
    h₁ : LE.le G.ediam 1
    ⊢ G.Adj u v
  -/
  rw [ediam_def, iSup_le_iff] at h₁
  /-
    case Adj.h.h.a
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Nontrivial α
    u v : α
    h₂ : LT.lt 0 (G.edist u v)
    h₁ : ∀ (i : Prod α α), LE.le (G.edist i.1 i.2) 1
    ⊢ G.Adj u v
  -/
  exact edist_eq_one_iff_adj.mp <| le_antisymm (h₁ (u, v)) <| Order.one_le_iff_pos.mpr h₂
  /-
    🎉 no goals
  -/


/--
The diameter is the greatest distance between any two vertices, with the value `0` in
case the distances are not bounded above, or the graph is not connected.
-/
noncomputable def diam (G : SimpleGraph α) :=
  G.ediam.toNat


lemma diam_def : G.diam = (⨆ p : α × α, G.edist p.1 p.2).toNat := by
  /-
    α : Type u_1
    G : SimpleGraph α
    ⊢ Eq G.diam (iSup fun p => G.edist p.1 p.2).toNat
  -/
  rw [diam, ediam_def]
  /-
    🎉 no goals
  -/


lemma dist_le_diam (h : G.ediam ≠ ⊤) {u v : α} : G.dist u v ≤ G.diam :=
  ENat.toNat_le_toNat edist_le_ediam h


lemma nontrivial_of_diam_ne_zero (h : G.diam ≠ 0) : Nontrivial α := by
  /-
    α : Type u_1
    G : SimpleGraph α
    h : Ne G.diam 0
    ⊢ Nontrivial α
  -/
  apply G.nontrivial_of_ediam_ne_zero
  /-
    α : Type u_1
    G : SimpleGraph α
    h : Ne G.diam 0
    ⊢ Ne G.ediam 0
  -/
  contrapose! h
  /-
    α : Type u_1
    G : SimpleGraph α
    h : Eq G.ediam 0
    ⊢ Eq G.diam 0
  -/
  simp [diam, h]
  /-
    🎉 no goals
  -/


lemma diam_eq_zero_of_not_connected (h : ¬G.Connected) : G.diam = 0 := by
  /-
    α : Type u_1
    G : SimpleGraph α
    h : Not G.Connected
    ⊢ Eq G.diam 0
  -/
  cases isEmpty_or_nonempty α
    /-
      case inl
      α : Type u_1
      G : SimpleGraph α
      h : Not G.Connected
      h✝ : IsEmpty α
      ⊢ Eq G.diam 0
    -/
  · rw [diam, ediam, ciSup_of_empty, bot_eq_zero']; rfl
                                                    /-
                                                      🎉 no goals
                                                    -/
    /-
      case inr
      α : Type u_1
      G : SimpleGraph α
      h : Not G.Connected
      h✝ : Nonempty α
      ⊢ Eq G.diam 0
    -/
  · rw [diam, ediam_eq_top_of_not_connected h, ENat.toNat_top]
    /-
      🎉 no goals
    -/


lemma diam_eq_zero_of_ediam_eq_top (h : G.ediam = ⊤) : G.diam = 0 := by
  /-
    α : Type u_1
    G : SimpleGraph α
    h : Eq G.ediam Top.top
    ⊢ Eq G.diam 0
  -/
  rw [diam, h, ENat.toNat_top]
  /-
    🎉 no goals
  -/


lemma ediam_ne_top_of_diam_ne_zero (h : G.diam ≠ 0) : G.ediam ≠ ⊤ :=
  mt diam_eq_zero_of_ediam_eq_top  h


lemma exists_dist_eq_diam [Nonempty α] :
    ∃ u v, G.dist u v = G.diam := by
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Nonempty α
    ⊢ Exists fun u => Exists fun v => Eq (G.dist u v) G.diam
  -/
  by_cases h : G.diam = 0
    /-
      case pos
      α : Type u_1
      G : SimpleGraph α
      inst✝ : Nonempty α
      h : Eq G.diam 0
      ⊢ Exists fun u => Exists fun v => Eq (G.dist u v) G.diam
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      G : SimpleGraph α
      inst✝ : Nonempty α
      h : Not (Eq G.diam 0)
      ⊢ Exists fun u => Exists fun v => Eq (G.dist u v) G.diam
    -/
  · obtain ⟨u, v, huv⟩ := exists_edist_eq_ediam_of_ne_top <| ediam_ne_top_of_diam_ne_zero h
    /-
      case neg.intro.intro
      α : Type u_1
      G : SimpleGraph α
      inst✝ : Nonempty α
      h : Not (Eq G.diam 0)
      u v : α
      huv : Eq (G.edist u v) G.ediam
      ⊢ Exists fun u => Exists fun v => Eq (G.dist u v) G.diam
    -/
    use u, v
    /-
      case h
      α : Type u_1
      G : SimpleGraph α
      inst✝ : Nonempty α
      h : Not (Eq G.diam 0)
      u v : α
      huv : Eq (G.edist u v) G.ediam
      ⊢ Eq (G.dist u v) G.diam
    -/
    rw [diam, dist, congrArg ENat.toNat huv]
    /-
      🎉 no goals
    -/


@[gcongr]
lemma diam_anti_of_ediam_ne_top (h : G ≤ G') (hn : G.ediam ≠ ⊤) : G'.diam ≤ G.diam :=
  ENat.toNat_le_toNat (ediam_anti h) hn


@[simp]
lemma diam_bot : (⊥ : SimpleGraph α).diam = 0 := by
  /-
    α : Type u_1
    ⊢ Eq Bot.bot.diam 0
  -/
  rw [diam, ENat.toNat_eq_zero]
  /-
    α : Type u_1
    ⊢ Or (Eq Bot.bot.ediam 0) (Eq Bot.bot.ediam Top.top)
  -/
  cases subsingleton_or_nontrivial α
    /-
      case inl
      α : Type u_1
      h✝ : Subsingleton α
      ⊢ Or (Eq Bot.bot.ediam 0) (Eq Bot.bot.ediam Top.top)
    -/
  · exact Or.inl ediam_eq_zero_of_subsingleton
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      h✝ : Nontrivial α
      ⊢ Or (Eq Bot.bot.ediam 0) (Eq Bot.bot.ediam Top.top)
    -/
  · exact Or.inr ediam_bot
    /-
      🎉 no goals
    -/


@[simp]
lemma diam_top [Nontrivial α] : (⊤ : SimpleGraph α).diam = 1 := by
  /-
    α : Type u_1
    inst✝ : Nontrivial α
    ⊢ Eq Top.top.diam 1
  -/
  rw [diam, ediam_top, ENat.toNat_one]
  /-
    🎉 no goals
  -/


@[simp]
lemma diam_eq_zero : G.diam = 0 ↔ G.ediam = ⊤ ∨ Subsingleton α := by
  /-
    α : Type u_1
    G : SimpleGraph α
    ⊢ Iff (Eq G.diam 0) (Or (Eq G.ediam Top.top) (Subsingleton α))
  -/
  rw [diam, ENat.toNat_eq_zero, or_comm, ediam_eq_zero_iff_subsingleton]
  /-
    🎉 no goals
  -/


@[simp]
lemma diam_eq_one [Nontrivial α] : G.diam = 1 ↔ G = ⊤ := by
  /-
    α : Type u_1
    G : SimpleGraph α
    inst✝ : Nontrivial α
    ⊢ Iff (Eq G.diam 1) (Eq G Top.top)
  -/
  rw [diam, ENat.toNat_eq_iff one_ne_zero, Nat.cast_one, ediam_eq_one]
  /-
    🎉 no goals
  -/



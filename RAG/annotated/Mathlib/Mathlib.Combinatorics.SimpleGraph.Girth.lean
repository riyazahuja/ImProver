/--
The extended girth of a simple graph is the length of its smallest cycle, or `∞` if the graph is
acyclic.
-/
noncomputable def egirth (G : SimpleGraph α) : ℕ∞ :=
  ⨅ a, ⨅ w : G.Walk a a, ⨅ _ : w.IsCycle, w.length


@[simp]
lemma le_egirth {n : ℕ∞} : n ≤ G.egirth ↔ ∀ a (w : G.Walk a a), w.IsCycle → n ≤ w.length := by
  /-
    α : Type u_1
    G : SimpleGraph α
    n : ENat
    ⊢ Iff (LE.le n G.egirth) (∀ (a : α) (w : G.Walk a a), w.IsCycle → LE.le n ↑w.l …
  -/
  simp [egirth]
  /-
    🎉 no goals
  -/


@[simp]
                                                       /-
                                                         α : Type u_1
                                                         G : SimpleGraph α
                                                         ⊢ Iff (Eq G.egirth Top.top) G.IsAcyclic
                                                       -/
lemma egirth_eq_top : G.egirth = ⊤ ↔ G.IsAcyclic := by simp [egirth, IsAcyclic]
                                                       /-
                                                         🎉 no goals
                                                       -/


protected alias ⟨_, IsAcyclic.egirth_eq_top⟩ := egirth_eq_top


lemma egirth_anti : Antitone (egirth : SimpleGraph α → ℕ∞) :=
                                                                                  /-
                                                                                    α : Type u_1
                                                                                    G H : SimpleGraph α
                                                                                    h : LE.le G H
                                                                                    a : α
                                                                                    w : G.Walk a a
                                                                                    hw : w.IsCycle
                                                                                    ⊢ LE.le ↑(SimpleGraph.Walk.mapLe h w).length ↑w.length
                                                                                  -/
  fun G H h ↦ iInf_mono fun a ↦ iInf₂_mono' fun w hw ↦ ⟨w.mapLe h, hw.mapLe _, by simp⟩
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


lemma exists_egirth_eq_length :
    (∃ (a : α) (w : G.Walk a a), w.IsCycle ∧ G.egirth = w.length) ↔ ¬ G.IsAcyclic := by
  /-
    α : Type u_1
    G : SimpleGraph α
    ⊢ Iff (Exists fun a => Exists fun w => And w.IsCycle (Eq G.egirth ↑w.length))  …
  -/
  refine ⟨?_, fun h ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      G : SimpleGraph α
      ⊢ (Exists fun a => Exists fun w => And w.IsCycle (Eq G.egirth ↑w.length)) → No …
    -/
  · rintro ⟨a, w, hw, _⟩ hG
    /-
      case refine_1.intro.intro.intro
      α : Type u_1
      G : SimpleGraph α
      a : α
      w : G.Walk a a
      hw : w.IsCycle
      right✝ : Eq G.egirth ↑w.length
      hG : G.IsAcyclic
      ⊢ False
    -/
    exact hG _ hw
    /-
      🎉 no goals
    -/
  · simp_rw [← egirth_eq_top, ← Ne.eq_def, egirth, iInf_subtype', iInf_sigma', ENat.iInf_coe_ne_top,
      ← exists_prop, Subtype.exists', Sigma.exists', eq_comm] at h ⊢
    /-
      case refine_2
      α : Type u_1
      G : SimpleGraph α
      h : Nonempty (Sigma fun i => Subtype SimpleGraph.Walk.IsCycle)
      ⊢ Exists fun x => Eq (↑(↑x.snd).length) (iInf fun x => ↑(↑x.snd).length)
    -/
    exact ciInf_mem _
    /-
      🎉 no goals
    -/


lemma three_le_egirth : 3 ≤ G.egirth := by
  /-
    α : Type u_1
    G : SimpleGraph α
    ⊢ LE.le 3 G.egirth
  -/
  by_cases h : G.IsAcyclic
    /-
      case pos
      α : Type u_1
      G : SimpleGraph α
      h : G.IsAcyclic
      ⊢ LE.le 3 G.egirth
    -/
  · rw [← egirth_eq_top] at h
    /-
      case pos
      α : Type u_1
      G : SimpleGraph α
      h : Eq G.egirth Top.top
      ⊢ LE.le 3 G.egirth
    -/
    rw [h]
    /-
      case pos
      α : Type u_1
      G : SimpleGraph α
      h : Eq G.egirth Top.top
      ⊢ LE.le 3 Top.top
    -/
    apply le_top
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      G : SimpleGraph α
      h : Not G.IsAcyclic
      ⊢ LE.le 3 G.egirth
    -/
  · rw [← exists_egirth_eq_length] at h
    /-
      case neg
      α : Type u_1
      G : SimpleGraph α
      h : Exists fun a => Exists fun w => And w.IsCycle (Eq G.egirth ↑w.length)
      ⊢ LE.le 3 G.egirth
    -/
    have ⟨_, _, _⟩ := h
    /-
      case neg
      α : Type u_1
      G : SimpleGraph α
      h : Exists fun a => Exists fun w => And w.IsCycle (Eq G.egirth ↑w.length)
      w✝¹ : α
      w✝ : G.Walk w✝¹ w✝¹
      h✝ : And w✝.IsCycle (Eq G.egirth ↑w✝.length)
      ⊢ LE.le 3 G.egirth
    -/
    simp_all only [Nat.cast_inj, Nat.ofNat_le_cast, Walk.IsCycle.three_le_length]
    /-
      🎉 no goals
    -/


                                                                /-
                                                                  α : Type u_1
                                                                  ⊢ Eq Bot.bot.egirth Top.top
                                                                -/
@[simp] lemma egirth_bot : egirth (⊥ : SimpleGraph α) = ⊤ := by simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


/--
The girth of a simple graph is the length of its smallest cycle, or junk value `0` if the graph is
acyclic.
-/
noncomputable def girth (G : SimpleGraph α) : ℕ :=
  G.egirth.toNat


lemma three_le_girth (hG : ¬ G.IsAcyclic) : 3 ≤ G.girth :=
  ENat.toNat_le_toNat three_le_egirth <| egirth_eq_top.not.mpr hG


lemma girth_eq_zero : G.girth = 0 ↔ G.IsAcyclic :=
                                                 /-
                                                   α : Type u_1
                                                   G : SimpleGraph α
                                                   h : Eq G.girth 0
                                                   ⊢ Not (LE.le 3 G.girth)
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  ⟨fun h ↦ not_not.mp <| three_le_girth.mt <| by omega, fun h ↦ by simp [girth, h]⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


protected alias ⟨_, IsAcyclic.girth_eq_zero⟩ := girth_eq_zero


lemma girth_anti {G' : SimpleGraph α} (hab : G ≤ G') (h : ¬ G.IsAcyclic) : G'.girth ≤ G.girth :=
  ENat.toNat_le_toNat (egirth_anti hab) <| egirth_eq_top.not.mpr h


lemma exists_girth_eq_length :
    (∃ (a : α) (w : G.Walk a a), w.IsCycle ∧ G.girth = w.length) ↔ ¬ G.IsAcyclic := by
  /-
    α : Type u_1
    G : SimpleGraph α
    ⊢ Iff (Exists fun a => Exists fun w => And w.IsCycle (Eq G.girth w.length)) (N …
  -/
  refine ⟨by tauto, fun h ↦ ?_⟩
  /-
    α : Type u_1
    G : SimpleGraph α
    h : Not G.IsAcyclic
    ⊢ Exists fun a => Exists fun w => And w.IsCycle (Eq G.girth w.length)
  -/
  obtain ⟨_, _, _⟩ := exists_egirth_eq_length.mpr h
  /-
    case intro.intro
    α : Type u_1
    G : SimpleGraph α
    h : Not G.IsAcyclic
    w✝¹ : α
    w✝ : G.Walk w✝¹ w✝¹
    h✝ : And w✝.IsCycle (Eq G.egirth ↑w✝.length)
    ⊢ Exists fun a => Exists fun w => And w.IsCycle (Eq G.girth w.length)
  -/
  simp_all only [girth, ENat.toNat_coe]
  /-
    case intro.intro
    α : Type u_1
    G : SimpleGraph α
    h : Not G.IsAcyclic
    w✝¹ : α
    w✝ : G.Walk w✝¹ w✝¹
    h✝ : And w✝.IsCycle (Eq G.egirth ↑w✝.length)
    ⊢ Exists fun a => Exists fun w => And w.IsCycle (Eq w✝.length w.length)
  -/
  tauto
  /-
    🎉 no goals
  -/


@[simp] lemma girth_bot : girth (⊥ : SimpleGraph α) = 0 := by
  /-
    α : Type u_1
    ⊢ Eq Bot.bot.girth 0
  -/
  simp [girth]
  /-
    🎉 no goals
  -/



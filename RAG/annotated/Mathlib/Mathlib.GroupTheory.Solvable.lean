/-- The derived series of the group `G`, obtained by starting from the subgroup `⊤` and repeatedly
  taking the commutator of the previous subgroup with itself for `n` times. -/
def derivedSeries : ℕ → Subgroup G
  | 0 => ⊤
  | n + 1 => ⁅derivedSeries n, derivedSeries n⁆


@[simp]
theorem derivedSeries_zero : derivedSeries G 0 = ⊤ :=
  rfl


@[simp]
theorem derivedSeries_succ (n : ℕ) :
    derivedSeries G (n + 1) = ⁅derivedSeries G n, derivedSeries G n⁆ :=
  rfl

-- Porting note: had to provide inductive hypothesis explicitly

theorem derivedSeries_normal (n : ℕ) : (derivedSeries G n).Normal := by
  induction n with
  | zero => exact (⊤ : Subgroup G).normal_of_characteristic
  | succ n ih => exact @Subgroup.commutator_normal G _ (derivedSeries G n) (derivedSeries G n) ih ih

-- Porting note: higher simp priority to restore Lean 3 behavior

@[simp 1100]
theorem derivedSeries_one : derivedSeries G 1 = commutator G :=
  rfl


theorem map_derivedSeries_le_derivedSeries (n : ℕ) :
    (derivedSeries G n).map f ≤ derivedSeries G' n := by
  induction n with
  | zero => exact le_top
  | succ n ih => simp only [derivedSeries_succ, map_commutator, commutator_mono, ih]


theorem derivedSeries_le_map_derivedSeries (hf : Function.Surjective f) (n : ℕ) :
    derivedSeries G' n ≤ (derivedSeries G n).map f := by
  induction n with
  | zero => exact (map_top_of_surjective f hf).ge
  | succ n ih => exact commutator_le_map_commutator ih ih


theorem map_derivedSeries_eq (hf : Function.Surjective f) (n : ℕ) :
    (derivedSeries G n).map f = derivedSeries G' n :=
  le_antisymm (map_derivedSeries_le_derivedSeries f n) (derivedSeries_le_map_derivedSeries hf n)


/-- A group `G` is solvable if its derived series is eventually trivial. We use this definition
  because it's the most convenient one to work with. -/
@[mk_iff isSolvable_def]
class IsSolvable : Prop where
  /-- A group `G` is solvable if its derived series is eventually trivial. -/
  solvable : ∃ n : ℕ, derivedSeries G n = ⊥


instance (priority := 100) CommGroup.isSolvable {G : Type*} [CommGroup G] : IsSolvable G :=
  ⟨⟨1, le_bot_iff.mp (Abelianization.commutator_subset_ker (MonoidHom.id G))⟩⟩


theorem isSolvable_of_comm {G : Type*} [hG : Group G] (h : ∀ a b : G, a * b = b * a) :
    IsSolvable G := by
  /-
    G : Type u_3
    hG : Group G
    h : ∀ (a b : G), Eq (HMul.hMul a b) (HMul.hMul b a)
    ⊢ IsSolvable G
  -/
  letI hG' : CommGroup G := { hG with mul_comm := h }
  /-
    G : Type u_3
    hG : Group G
    h : ∀ (a b : G), Eq (HMul.hMul a b) (HMul.hMul b a)
    hG' : CommGroup G := CommGroup.mk h
    ⊢ IsSolvable G
  -/
  cases hG
  /-
    case mk
    G : Type u_3
    toDivInvMonoid✝ : DivInvMonoid G
    inv_mul_cancel✝ : ∀ (a : G), Eq (HMul.hMul (Inv.inv a) a) 1
    h : ∀ (a b : G), Eq (HMul.hMul a b) (HMul.hMul b a)
    hG' : CommGroup G := CommGroup.mk h
    ⊢ IsSolvable G
  -/
  exact CommGroup.isSolvable
  /-
    🎉 no goals
  -/


theorem isSolvable_of_top_eq_bot (h : (⊤ : Subgroup G) = ⊥) : IsSolvable G :=
  ⟨⟨0, h⟩⟩


instance (priority := 100) isSolvable_of_subsingleton [Subsingleton G] : IsSolvable G :=
                                 /-
                                   G : Type u_1
                                   G' : Type u_2
                                   inst✝² : Group G
                                   inst✝¹ : Group G'
                                   f : MonoidHom G G'
                                   inst✝ : Subsingleton G
                                   ⊢ Eq Top.top Bot.bot
                                 -/
  isSolvable_of_top_eq_bot G (by simp [eq_iff_true_of_subsingleton])
                                 /-
                                   🎉 no goals
                                 -/


theorem solvable_of_ker_le_range {G' G'' : Type*} [Group G'] [Group G''] (f : G' →* G)
    (g : G →* G'') (hfg : g.ker ≤ f.range) [hG' : IsSolvable G'] [hG'' : IsSolvable G''] :
    IsSolvable G := by
  /-
    G : Type u_1
    inst✝² : Group G
    G' : Type u_3
    G'' : Type u_4
    inst✝¹ : Group G'
    inst✝ : Group G''
    f : MonoidHom G' G
    g : MonoidHom G G''
    hfg : LE.le g.ker f.range
    hG' : IsSolvable G'
    hG'' : IsSolvable G''
    ⊢ IsSolvable G
  -/
  obtain ⟨n, hn⟩ := id hG''
  /-
    case mk.intro
    G : Type u_1
    inst✝² : Group G
    G' : Type u_3
    G'' : Type u_4
    inst✝¹ : Group G'
    inst✝ : Group G''
    f : MonoidHom G' G
    g : MonoidHom G G''
    hfg : LE.le g.ker f.range
    hG' : IsSolvable G'
    hG'' : IsSolvable G''
    n : Nat
    hn : Eq (derivedSeries G'' n) Bot.bot
    ⊢ IsSolvable G
  -/
  obtain ⟨m, hm⟩ := id hG'
  /-
    case mk.intro.mk.intro
    G : Type u_1
    inst✝² : Group G
    G' : Type u_3
    G'' : Type u_4
    inst✝¹ : Group G'
    inst✝ : Group G''
    f : MonoidHom G' G
    g : MonoidHom G G''
    hfg : LE.le g.ker f.range
    hG' : IsSolvable G'
    hG'' : IsSolvable G''
    n : Nat
    hn : Eq (derivedSeries G'' n) Bot.bot
    m : Nat
    hm : Eq (derivedSeries G' m) Bot.bot
    ⊢ IsSolvable G
  -/
  refine ⟨⟨n + m, le_bot_iff.mp (Subgroup.map_bot f ▸ hm ▸ ?_)⟩⟩
  /-
    case mk.intro.mk.intro
    G : Type u_1
    inst✝² : Group G
    G' : Type u_3
    G'' : Type u_4
    inst✝¹ : Group G'
    inst✝ : Group G''
    f : MonoidHom G' G
    g : MonoidHom G G''
    hfg : LE.le g.ker f.range
    hG' : IsSolvable G'
    hG'' : IsSolvable G''
    n : Nat
    hn : Eq (derivedSeries G'' n) Bot.bot
    m : Nat
    hm : Eq (derivedSeries G' m) Bot.bot
    ⊢ LE.le (derivedSeries G (HAdd.hAdd n m)) (Subgroup.map f (derivedSeries G' m))
  -/
  clear hm
  /-
    case mk.intro.mk.intro
    G : Type u_1
    inst✝² : Group G
    G' : Type u_3
    G'' : Type u_4
    inst✝¹ : Group G'
    inst✝ : Group G''
    f : MonoidHom G' G
    g : MonoidHom G G''
    hfg : LE.le g.ker f.range
    hG' : IsSolvable G'
    hG'' : IsSolvable G''
    n : Nat
    hn : Eq (derivedSeries G'' n) Bot.bot
    m : Nat
    ⊢ LE.le (derivedSeries G (HAdd.hAdd n m)) (Subgroup.map f (derivedSeries G' m))
  -/
  induction' m with m hm
  · exact f.range_eq_map ▸ ((derivedSeries G n).map_eq_bot_iff.mp
      (le_bot_iff.mp ((map_derivedSeries_le_derivedSeries g n).trans hn.le))).trans hfg
    /-
      case mk.intro.mk.intro.succ
      G : Type u_1
      inst✝² : Group G
      G' : Type u_3
      G'' : Type u_4
      inst✝¹ : Group G'
      inst✝ : Group G''
      f : MonoidHom G' G
      g : MonoidHom G G''
      hfg : LE.le g.ker f.range
      hG' : IsSolvable G'
      hG'' : IsSolvable G''
      n : Nat
      hn : Eq (derivedSeries G'' n) Bot.bot
      m : Nat
      hm : LE.le (derivedSeries G (HAdd.hAdd n m)) (Subgroup.map f (derivedSeries G' …
      ⊢ LE.le (derivedSeries G (HAdd.hAdd n (HAdd.hAdd m 1))) (Subgroup.map f (deriv …
    -/
  · exact commutator_le_map_commutator hm hm
    /-
      🎉 no goals
    -/


theorem solvable_of_solvable_injective (hf : Function.Injective f) [IsSolvable G'] :
    IsSolvable G :=
  solvable_of_ker_le_range (1 : G' →* G) f ((f.ker_eq_bot_iff.mpr hf).symm ▸ bot_le)


instance subgroup_solvable_of_solvable (H : Subgroup G) [IsSolvable G] : IsSolvable H :=
  solvable_of_solvable_injective H.subtype_injective


theorem solvable_of_surjective (hf : Function.Surjective f) [IsSolvable G] : IsSolvable G' :=
  solvable_of_ker_le_range f (1 : G' →* G) (f.range_eq_top_of_surjective hf ▸ le_top)


instance solvable_quotient_of_solvable (H : Subgroup G) [H.Normal] [IsSolvable G] :
    IsSolvable (G ⧸ H) :=
  solvable_of_surjective (QuotientGroup.mk'_surjective H)


instance solvable_prod {G' : Type*} [Group G'] [IsSolvable G] [IsSolvable G'] :
    IsSolvable (G × G') :=
  solvable_of_ker_le_range (MonoidHom.inl G G') (MonoidHom.snd G G') fun x hx =>
    ⟨x.1, Prod.ext rfl hx.symm⟩


variable (G) in
theorem IsSolvable.commutator_lt_top_of_nontrivial [hG : IsSolvable G] [Nontrivial G] :
    commutator G < ⊤ := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    hG : IsSolvable G
    inst✝ : Nontrivial G
    ⊢ LT.lt (commutator G) Top.top
  -/
  rw [lt_top_iff_ne_top]
  /-
    G : Type u_1
    inst✝¹ : Group G
    hG : IsSolvable G
    inst✝ : Nontrivial G
    ⊢ Ne (commutator G) Top.top
  -/
  obtain ⟨n, hn⟩ := hG
  /-
    case mk.intro
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Nontrivial G
    n : Nat
    hn : Eq (derivedSeries G n) Bot.bot
    ⊢ Ne (commutator G) Top.top
  -/
  contrapose! hn
  /-
    case mk.intro
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Nontrivial G
    n : Nat
    hn : Eq (commutator G) Top.top
    ⊢ Ne (derivedSeries G n) Bot.bot
  -/
  refine ne_of_eq_of_ne ?_ top_ne_bot
  /-
    case mk.intro
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Nontrivial G
    n : Nat
    hn : Eq (commutator G) Top.top
    ⊢ Eq (derivedSeries G n) Top.top
  -/
  induction' n with n h
    /-
      case mk.intro.zero
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : Nontrivial G
      hn : Eq (commutator G) Top.top
      ⊢ Eq (derivedSeries G 0) Top.top
    -/
  · exact derivedSeries_zero G
    /-
      🎉 no goals
    -/
    /-
      case mk.intro.succ
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : Nontrivial G
      hn : Eq (commutator G) Top.top
      n : Nat
      h : Eq (derivedSeries G n) Top.top
      ⊢ Eq (derivedSeries G (HAdd.hAdd n 1)) Top.top
    -/
  · rwa [derivedSeries_succ, h]
    /-
      🎉 no goals
    -/


theorem IsSolvable.commutator_lt_of_ne_bot [IsSolvable G] {H : Subgroup G} (hH : H ≠ ⊥) :
    ⁅H, H⁆ < H := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : IsSolvable G
    H : Subgroup G
    hH : Ne H Bot.bot
    ⊢ LT.lt (Bracket.bracket H H) H
  -/
  rw [← nontrivial_iff_ne_bot] at hH
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : IsSolvable G
    H : Subgroup G
    hH : Nontrivial (Subtype fun x => Membership.mem H x)
    ⊢ LT.lt (Bracket.bracket H H) H
  -/
  rw [← H.range_subtype, MonoidHom.range_eq_map, ← map_commutator, map_subtype_lt_map_subtype]
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : IsSolvable G
    H : Subgroup G
    hH : Nontrivial (Subtype fun x => Membership.mem H x)
    ⊢ LT.lt (Bracket.bracket Top.top Top.top) Top.top
  -/
  exact commutator_lt_top_of_nontrivial H
  /-
    🎉 no goals
  -/


theorem isSolvable_iff_commutator_lt [WellFoundedLT (Subgroup G)] :
    IsSolvable G ↔ ∀ H : Subgroup G, H ≠ ⊥ → ⁅H, H⁆ < H := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : WellFoundedLT (Subgroup G)
    ⊢ Iff (IsSolvable G) (∀ (H : Subgroup G), Ne H Bot.bot → LT.lt (Bracket.bracke …
  -/
  refine ⟨fun _ _ ↦ IsSolvable.commutator_lt_of_ne_bot, fun h ↦ ?_⟩
  suffices h : IsSolvable (⊤ : Subgroup G) from
    solvable_of_surjective (MonoidHom.range_eq_top.mp (range_subtype ⊤))
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : WellFoundedLT (Subgroup G)
    h : ∀ (H : Subgroup G), Ne H Bot.bot → LT.lt (Bracket.bracket H H) H
    ⊢ IsSolvable (Subtype fun x => Membership.mem Top.top x)
  -/
  refine WellFoundedLT.induction (C := fun (H : Subgroup G) ↦ IsSolvable H) ⊤ fun H hH ↦ ?_
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : WellFoundedLT (Subgroup G)
    h : ∀ (H : Subgroup G), Ne H Bot.bot → LT.lt (Bracket.bracket H H) H
    H : Subgroup G
    hH : ∀ (y : Subgroup G), LT.lt y H → (fun H => IsSolvable (Subtype fun x => Me …
    ⊢ (fun H => IsSolvable (Subtype fun x => Membership.mem H x)) H
  -/
  rcases eq_or_ne H ⊥ with rfl | h'
    /-
      case inl
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : WellFoundedLT (Subgroup G)
      h : ∀ (H : Subgroup G), Ne H Bot.bot → LT.lt (Bracket.bracket H H) H
      hH : ∀ (y : Subgroup G), LT.lt y Bot.bot → (fun H => IsSolvable (Subtype fun x …
      ⊢ IsSolvable (Subtype fun x => Membership.mem Bot.bot x)
    -/
  · infer_instance
    /-
      🎉 no goals
    -/
    /-
      case inr
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : WellFoundedLT (Subgroup G)
      h : ∀ (H : Subgroup G), Ne H Bot.bot → LT.lt (Bracket.bracket H H) H
      H : Subgroup G
      hH : ∀ (y : Subgroup G), LT.lt y H → (fun H => IsSolvable (Subtype fun x => Me …
      h' : Ne H Bot.bot
      ⊢ IsSolvable (Subtype fun x => Membership.mem H x)
    -/
  · obtain ⟨n, hn⟩ := hH ⁅H, H⁆ (h H h')
    /-
      case inr.mk.intro
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : WellFoundedLT (Subgroup G)
      h : ∀ (H : Subgroup G), Ne H Bot.bot → LT.lt (Bracket.bracket H H) H
      H : Subgroup G
      hH : ∀ (y : Subgroup G), LT.lt y H → (fun H => IsSolvable (Subtype fun x => Me …
      h' : Ne H Bot.bot
      n : Nat
      hn : Eq (derivedSeries (Subtype fun x => Membership.mem (Bracket.bracket H H)  …
      ⊢ IsSolvable (Subtype fun x => Membership.mem H x)
    -/
    use n + 1
    /-
      case h
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : WellFoundedLT (Subgroup G)
      h : ∀ (H : Subgroup G), Ne H Bot.bot → LT.lt (Bracket.bracket H H) H
      H : Subgroup G
      hH : ∀ (y : Subgroup G), LT.lt y H → (fun H => IsSolvable (Subtype fun x => Me …
      h' : Ne H Bot.bot
      n : Nat
      hn : Eq (derivedSeries (Subtype fun x => Membership.mem (Bracket.bracket H H)  …
      ⊢ Eq (derivedSeries (Subtype fun x => Membership.mem H x) (HAdd.hAdd n 1)) Bot …
    -/
    rw [← (map_injective (subtype_injective _)).eq_iff, Subgroup.map_bot] at hn ⊢
    /-
      case h
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : WellFoundedLT (Subgroup G)
      h : ∀ (H : Subgroup G), Ne H Bot.bot → LT.lt (Bracket.bracket H H) H
      H : Subgroup G
      hH : ∀ (y : Subgroup G), LT.lt y H → (fun H => IsSolvable (Subtype fun x => Me …
      h' : Ne H Bot.bot
      n : Nat
      hn : Eq (Subgroup.map (Bracket.bracket H H).subtype (derivedSeries (Subtype fu …
      ⊢ Eq (Subgroup.map H.subtype (derivedSeries (Subtype fun x => Membership.mem H …
    -/
    rw [← hn]
    /-
      case h
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : WellFoundedLT (Subgroup G)
      h : ∀ (H : Subgroup G), Ne H Bot.bot → LT.lt (Bracket.bracket H H) H
      H : Subgroup G
      hH : ∀ (y : Subgroup G), LT.lt y H → (fun H => IsSolvable (Subtype fun x => Me …
      h' : Ne H Bot.bot
      n : Nat
      hn : Eq (Subgroup.map (Bracket.bracket H H).subtype (derivedSeries (Subtype fu …
      ⊢ Eq (Subgroup.map H.subtype (derivedSeries (Subtype fun x => Membership.mem H …
    -/
    clear hn
    /-
      case h
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : WellFoundedLT (Subgroup G)
      h : ∀ (H : Subgroup G), Ne H Bot.bot → LT.lt (Bracket.bracket H H) H
      H : Subgroup G
      hH : ∀ (y : Subgroup G), LT.lt y H → (fun H => IsSolvable (Subtype fun x => Me …
      h' : Ne H Bot.bot
      n : Nat
      ⊢ Eq (Subgroup.map H.subtype (derivedSeries (Subtype fun x => Membership.mem H …
    -/
    induction' n with n ih
    · rw [derivedSeries_succ, derivedSeries_zero, derivedSeries_zero, map_commutator,
        ← MonoidHom.range_eq_map, ← MonoidHom.range_eq_map, range_subtype, range_subtype]
      /-
        case h.succ
        G : Type u_1
        inst✝¹ : Group G
        inst✝ : WellFoundedLT (Subgroup G)
        h : ∀ (H : Subgroup G), Ne H Bot.bot → LT.lt (Bracket.bracket H H) H
        H : Subgroup G
        hH : ∀ (y : Subgroup G), LT.lt y H → (fun H => IsSolvable (Subtype fun x => Me …
        h' : Ne H Bot.bot
        n : Nat
        ih : Eq (Subgroup.map H.subtype (derivedSeries (Subtype fun x => Membership.me …
        ⊢ Eq (Subgroup.map H.subtype (derivedSeries (Subtype fun x => Membership.mem H …
      -/
    · rw [derivedSeries_succ, map_commutator, ih, derivedSeries_succ, map_commutator]
      /-
        🎉 no goals
      -/


theorem IsSimpleGroup.derivedSeries_succ {n : ℕ} : derivedSeries G n.succ = commutator G := by
  induction n with
  | zero => exact derivedSeries_one G
  | succ n ih =>
    rw [_root_.derivedSeries_succ, ih, _root_.commutator]
    cases' (commutator_normal (⊤ : Subgroup G) (⊤ : Subgroup G)).eq_bot_or_eq_top with h h
    · rw [h, commutator_bot_left]
    · rwa [h]


theorem IsSimpleGroup.comm_iff_isSolvable : (∀ a b : G, a * b = b * a) ↔ IsSolvable G :=
  ⟨isSolvable_of_comm, fun ⟨⟨n, hn⟩⟩ => by
    /-
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : IsSimpleGroup G
      x✝ : IsSolvable G
      n : Nat
      hn : Eq (derivedSeries G n) Bot.bot
      ⊢ ∀ (a b : G), Eq (HMul.hMul a b) (HMul.hMul b a)
    -/
    cases n
      /-
        case zero
        G : Type u_1
        inst✝¹ : Group G
        inst✝ : IsSimpleGroup G
        x✝ : IsSolvable G
        hn : Eq (derivedSeries G 0) Bot.bot
        ⊢ ∀ (a b : G), Eq (HMul.hMul a b) (HMul.hMul b a)
      -/
    · intro a b
      /-
        case zero
        G : Type u_1
        inst✝¹ : Group G
        inst✝ : IsSimpleGroup G
        x✝ : IsSolvable G
        hn : Eq (derivedSeries G 0) Bot.bot
        a b : G
        ⊢ Eq (HMul.hMul a b) (HMul.hMul b a)
      -/
      refine (mem_bot.1 ?_).trans (mem_bot.1 ?_).symm <;>
          /-
            case zero.refine_1
            G : Type u_1
            inst✝¹ : Group G
            inst✝ : IsSimpleGroup G
            x✝ : IsSolvable G
            hn : Eq (derivedSeries G 0) Bot.bot
            a b : G
            ⊢ Membership.mem Bot.bot (HMul.hMul a b)
          -/
          /-
            case zero.refine_1
            G : Type u_1
            inst✝¹ : Group G
            inst✝ : IsSimpleGroup G
            x✝ : IsSolvable G
            hn : Eq (derivedSeries G 0) Bot.bot
            a b : G
            ⊢ Membership.mem (derivedSeries G 0) (HMul.hMul a b)
          -/
          /-
            🎉 no goals
          -/
          /-
            case zero.refine_2
            G : Type u_1
            inst✝¹ : Group G
            inst✝ : IsSimpleGroup G
            x✝ : IsSolvable G
            hn : Eq (derivedSeries G 0) Bot.bot
            a b : G
            ⊢ Membership.mem (derivedSeries G 0) (HMul.hMul b a)
          -/
          exact mem_top _
          /-
            🎉 no goals
          -/
      /-
        case succ
        G : Type u_1
        inst✝¹ : Group G
        inst✝ : IsSimpleGroup G
        x✝ : IsSolvable G
        n✝ : Nat
        hn : Eq (derivedSeries G (HAdd.hAdd n✝ 1)) Bot.bot
        ⊢ ∀ (a b : G), Eq (HMul.hMul a b) (HMul.hMul b a)
      -/
    · rw [IsSimpleGroup.derivedSeries_succ] at hn
      /-
        case succ
        G : Type u_1
        inst✝¹ : Group G
        inst✝ : IsSimpleGroup G
        x✝ : IsSolvable G
        n✝ : Nat
        hn : Eq (commutator G) Bot.bot
        ⊢ ∀ (a b : G), Eq (HMul.hMul a b) (HMul.hMul b a)
      -/
      intro a b
      /-
        case succ
        G : Type u_1
        inst✝¹ : Group G
        inst✝ : IsSimpleGroup G
        x✝ : IsSolvable G
        n✝ : Nat
        hn : Eq (commutator G) Bot.bot
        a b : G
        ⊢ Eq (HMul.hMul a b) (HMul.hMul b a)
      -/
      rw [← mul_inv_eq_one, mul_inv_rev, ← mul_assoc, ← mem_bot, ← hn, commutator_eq_closure]
      /-
        case succ
        G : Type u_1
        inst✝¹ : Group G
        inst✝ : IsSimpleGroup G
        x✝ : IsSolvable G
        n✝ : Nat
        hn : Eq (commutator G) Bot.bot
        a b : G
        ⊢ Membership.mem (Subgroup.closure (commutatorSet G)) (HMul.hMul (HMul.hMul (H …
      -/
      exact subset_closure ⟨a, b, rfl⟩⟩
      /-
        🎉 no goals
      -/


theorem not_solvable_of_mem_derivedSeries {g : G} (h1 : g ≠ 1)
    (h2 : ∀ n : ℕ, g ∈ derivedSeries G n) : ¬IsSolvable G :=
  mt (isSolvable_def _).mp
    (not_exists_of_forall_not fun n h =>
      h1 (Subgroup.mem_bot.mp ((congr_arg (g ∈ ·) h).mp (h2 n))))


theorem Equiv.Perm.fin_5_not_solvable : ¬IsSolvable (Equiv.Perm (Fin 5)) := by
  /-
    ⊢ Not (IsSolvable (Equiv.Perm (Fin 5)))
  -/
  let x : Equiv.Perm (Fin 5) := ⟨![1, 2, 0, 3, 4], ![2, 0, 1, 3, 4], by decide, by decide⟩
  /-
    x : Equiv.Perm (Fin 5) := { toFun := Matrix.vecCons 1 (Matrix.vecCons 2 (Matri …
    ⊢ Not (IsSolvable (Equiv.Perm (Fin 5)))
  -/
  let y : Equiv.Perm (Fin 5) := ⟨![3, 4, 2, 0, 1], ![3, 4, 2, 0, 1], by decide, by decide⟩
  /-
    x : Equiv.Perm (Fin 5) := { toFun := Matrix.vecCons 1 (Matrix.vecCons 2 (Matri …
    y : Equiv.Perm (Fin 5) := { toFun := Matrix.vecCons 3 (Matrix.vecCons 4 (Matri …
    ⊢ Not (IsSolvable (Equiv.Perm (Fin 5)))
  -/
  let z : Equiv.Perm (Fin 5) := ⟨![0, 3, 2, 1, 4], ![0, 3, 2, 1, 4], by decide, by decide⟩
  /-
    x : Equiv.Perm (Fin 5) := { toFun := Matrix.vecCons 1 (Matrix.vecCons 2 (Matri …
    y : Equiv.Perm (Fin 5) := { toFun := Matrix.vecCons 3 (Matrix.vecCons 4 (Matri …
    z : Equiv.Perm (Fin 5) := { toFun := Matrix.vecCons 0 (Matrix.vecCons 3 (Matri …
    ⊢ Not (IsSolvable (Equiv.Perm (Fin 5)))
  -/
  have key : x = z * ⁅x, y * x * y⁻¹⁆ * z⁻¹ := by unfold x y z; decide
  /-
    x : Equiv.Perm (Fin 5) := { toFun := Matrix.vecCons 1 (Matrix.vecCons 2 (Matri …
    y : Equiv.Perm (Fin 5) := { toFun := Matrix.vecCons 3 (Matrix.vecCons 4 (Matri …
    z : Equiv.Perm (Fin 5) := { toFun := Matrix.vecCons 0 (Matrix.vecCons 3 (Matri …
    key : Eq x (HMul.hMul (HMul.hMul z (Bracket.bracket x (HMul.hMul (HMul.hMul y  …
    ⊢ Not (IsSolvable (Equiv.Perm (Fin 5)))
  -/
  refine not_solvable_of_mem_derivedSeries (show x ≠ 1 by decide) fun n => ?_
  induction n with
  | zero => exact mem_top x
  | succ n ih =>
    rw [key, (derivedSeries_normal _ _).mem_comm_iff, inv_mul_cancel_left]
    exact commutator_mem_commutator ih ((derivedSeries_normal _ _).conj_mem _ ih _)


theorem Equiv.Perm.not_solvable (X : Type*) (hX : 5 ≤ Cardinal.mk X) :
    ¬IsSolvable (Equiv.Perm X) := by
  /-
    X : Type u_3
    hX : LE.le 5 (Cardinal.mk X)
    ⊢ Not (IsSolvable (Equiv.Perm X))
  -/
  intro h
  have key : Nonempty (Fin 5 ↪ X) := by
    rwa [← Cardinal.lift_mk_le, Cardinal.mk_fin, Cardinal.lift_natCast, Cardinal.lift_id]
  exact
    Equiv.Perm.fin_5_not_solvable
      (solvable_of_solvable_injective (Equiv.Perm.viaEmbeddingHom_injective (Nonempty.some key)))



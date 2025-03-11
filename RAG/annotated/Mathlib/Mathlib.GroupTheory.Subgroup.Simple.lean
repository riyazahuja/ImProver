/-- A `Group` is simple when it has exactly two normal `Subgroup`s. -/
class IsSimpleGroup extends Nontrivial G : Prop where
  /-- Any normal subgroup is either `⊥` or `⊤` -/
  eq_bot_or_eq_top_of_normal : ∀ H : Subgroup G, H.Normal → H = ⊥ ∨ H = ⊤


/-- An `AddGroup` is simple when it has exactly two normal `AddSubgroup`s. -/
class IsSimpleAddGroup extends Nontrivial A : Prop where
  /-- Any normal additive subgroup is either `⊥` or `⊤` -/
  eq_bot_or_eq_top_of_normal : ∀ H : AddSubgroup A, H.Normal → H = ⊥ ∨ H = ⊤


@[to_additive]
theorem Subgroup.Normal.eq_bot_or_eq_top [IsSimpleGroup G] {H : Subgroup G} (Hn : H.Normal) :
    H = ⊥ ∨ H = ⊤ :=
  IsSimpleGroup.eq_bot_or_eq_top_of_normal H Hn


@[to_additive]
instance {C : Type*} [CommGroup C] [IsSimpleGroup C] : IsSimpleOrder (Subgroup C) :=
  ⟨fun H => H.normal_of_comm.eq_bot_or_eq_top⟩


@[to_additive]
theorem isSimpleGroup_of_surjective {H : Type*} [Group H] [IsSimpleGroup G] [Nontrivial H]
    (f : G →* H) (hf : Function.Surjective f) : IsSimpleGroup H :=
  ⟨fun H iH => by
    /-
      G : Type u_1
      inst✝³ : Group G
      H✝ : Type u_3
      inst✝² : Group H✝
      inst✝¹ : IsSimpleGroup G
      inst✝ : Nontrivial H✝
      f : MonoidHom G H✝
      hf : Function.Surjective ⇑f
      H : Subgroup H✝
      iH : H.Normal
      ⊢ Or (Eq H Bot.bot) (Eq H Top.top)
    -/
    refine (iH.comap f).eq_bot_or_eq_top.imp (fun h => ?_) fun h => ?_
      /-
        case refine_1
        G : Type u_1
        inst✝³ : Group G
        H✝ : Type u_3
        inst✝² : Group H✝
        inst✝¹ : IsSimpleGroup G
        inst✝ : Nontrivial H✝
        f : MonoidHom G H✝
        hf : Function.Surjective ⇑f
        H : Subgroup H✝
        iH : H.Normal
        h : Eq (Subgroup.comap f H) Bot.bot
        ⊢ Eq H Bot.bot
      -/
    · rw [← map_bot f, ← h, map_comap_eq_self_of_surjective hf]
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        G : Type u_1
        inst✝³ : Group G
        H✝ : Type u_3
        inst✝² : Group H✝
        inst✝¹ : IsSimpleGroup G
        inst✝ : Nontrivial H✝
        f : MonoidHom G H✝
        hf : Function.Surjective ⇑f
        H : Subgroup H✝
        iH : H.Normal
        h : Eq (Subgroup.comap f H) Top.top
        ⊢ Eq H Top.top
      -/
    · rw [← comap_top f] at h
      /-
        case refine_2
        G : Type u_1
        inst✝³ : Group G
        H✝ : Type u_3
        inst✝² : Group H✝
        inst✝¹ : IsSimpleGroup G
        inst✝ : Nontrivial H✝
        f : MonoidHom G H✝
        hf : Function.Surjective ⇑f
        H : Subgroup H✝
        iH : H.Normal
        h : Eq (Subgroup.comap f H) (Subgroup.comap f Top.top)
        ⊢ Eq H Top.top
      -/
      exact comap_injective hf h⟩
      /-
        🎉 no goals
      -/



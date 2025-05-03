/-- Subgroups of a group `G` are isomorphic to additive subgroups of `Additive G`. -/
@[simps!]
def Subgroup.toAddSubgroup : Subgroup G ≃o AddSubgroup (Additive G) where
  toFun S := { Submonoid.toAddSubmonoid S.toSubmonoid with neg_mem' := S.inv_mem' }
  invFun S := { AddSubmonoid.toSubmonoid S.toAddSubmonoid with inv_mem' := S.neg_mem' }
                   /-
                     G : Type u_1
                     inst✝¹ : Group G
                     A : Type u_2
                     inst✝ : AddGroup A
                     x : Subgroup G
                     ⊢ Eq
                         ((fun S =>
                             let __src := AddSubmonoid.toSubmonoid S.toAddSubmonoid;
                             { toSubmonoid := __src, inv_mem' := ⋯ })
                           ((fun S =>
                               let __src := Submonoid.toAddSubmonoid S.toSubmonoid;
                               { toAddSubmonoid := __src, neg_mem' := ⋯ })
                             x))
                         x
                   -/
  left_inv x := by cases x; rfl
                            /-
                              🎉 no goals
                            -/
                    /-
                      G : Type u_1
                      inst✝¹ : Group G
                      A : Type u_2
                      inst✝ : AddGroup A
                      x : AddSubgroup (Additive G)
                      ⊢ Eq
                          ((fun S =>
                              let __src := Submonoid.toAddSubmonoid S.toSubmonoid;
                              { toAddSubmonoid := __src, neg_mem' := ⋯ })
                            ((fun S =>
                                let __src := AddSubmonoid.toSubmonoid S.toAddSubmonoid;
                                { toSubmonoid := __src, inv_mem' := ⋯ })
                              x))
                          x
                    -/
  right_inv x := by cases x; rfl
                             /-
                               🎉 no goals
                             -/
  map_rel_iff' := Iff.rfl


/-- Additive subgroup of an additive group `Additive G` are isomorphic to subgroup of `G`. -/
abbrev AddSubgroup.toSubgroup' : AddSubgroup (Additive G) ≃o Subgroup G :=
  Subgroup.toAddSubgroup.symm


/-- Additive subgroups of an additive group `A` are isomorphic to subgroups of `Multiplicative A`.
-/
@[simps!]
def AddSubgroup.toSubgroup : AddSubgroup A ≃o Subgroup (Multiplicative A) where
  toFun S := { AddSubmonoid.toSubmonoid S.toAddSubmonoid with inv_mem' := S.neg_mem' }
  invFun S := { Submonoid.toAddSubmonoid S.toSubmonoid with neg_mem' := S.inv_mem' }
                   /-
                     G : Type u_1
                     inst✝¹ : Group G
                     A : Type u_2
                     inst✝ : AddGroup A
                     x : AddSubgroup A
                     ⊢ Eq
                         ((fun S =>
                             let __src := Submonoid.toAddSubmonoid S.toSubmonoid;
                             { toAddSubmonoid := __src, neg_mem' := ⋯ })
                           ((fun S =>
                               let __src := AddSubmonoid.toSubmonoid S.toAddSubmonoid;
                               { toSubmonoid := __src, inv_mem' := ⋯ })
                             x))
                         x
                   -/
  left_inv x := by cases x; rfl
                            /-
                              🎉 no goals
                            -/
                    /-
                      G : Type u_1
                      inst✝¹ : Group G
                      A : Type u_2
                      inst✝ : AddGroup A
                      x : Subgroup (Multiplicative A)
                      ⊢ Eq
                          ((fun S =>
                              let __src := AddSubmonoid.toSubmonoid S.toAddSubmonoid;
                              { toSubmonoid := __src, inv_mem' := ⋯ })
                            ((fun S =>
                                let __src := Submonoid.toAddSubmonoid S.toSubmonoid;
                                { toAddSubmonoid := __src, neg_mem' := ⋯ })
                              x))
                          x
                    -/
  right_inv x := by cases x; rfl
                             /-
                               🎉 no goals
                             -/
  map_rel_iff' := Iff.rfl


/-- Subgroups of an additive group `Multiplicative A` are isomorphic to additive subgroups of `A`.
-/
abbrev Subgroup.toAddSubgroup' : Subgroup (Multiplicative A) ≃o AddSubgroup A :=
  AddSubgroup.toSubgroup.symm


/-- The subgroup `G` of the group `G`. -/
@[to_additive "The `AddSubgroup G` of the `AddGroup G`."]
instance : Top (Subgroup G) :=
  ⟨{ (⊤ : Submonoid G) with inv_mem' := fun _ => Set.mem_univ _ }⟩


/-- The top subgroup is isomorphic to the group.

This is the group version of `Submonoid.topEquiv`. -/
@[to_additive (attr := simps!)
      "The top additive subgroup is isomorphic to the additive group.

      This is the additive group version of `AddSubmonoid.topEquiv`."]
def topEquiv : (⊤ : Subgroup G) ≃* G :=
  Submonoid.topEquiv


/-- The trivial subgroup `{1}` of a group `G`. -/
@[to_additive "The trivial `AddSubgroup` `{0}` of an `AddGroup` `G`."]
instance : Bot (Subgroup G) :=
                                           /-
                                             G : Type u_1
                                             inst✝ : Group G
                                             H K : Subgroup G
                                             ⊢ ∀ {x : G}, Membership.mem __src✝.carrier x → Membership.mem __src✝.carrier ( …
                                           -/
  ⟨{ (⊥ : Submonoid G) with inv_mem' := by simp}⟩
                                           /-
                                             🎉 no goals
                                           -/


@[to_additive]
instance : Inhabited (Subgroup G) :=
  ⟨⊥⟩


@[to_additive (attr := simp)]
theorem mem_bot {x : G} : x ∈ (⊥ : Subgroup G) ↔ x = 1 :=
  Iff.rfl


@[to_additive (attr := simp)]
theorem mem_top (x : G) : x ∈ (⊤ : Subgroup G) :=
  Set.mem_univ x


@[to_additive (attr := simp)]
theorem coe_top : ((⊤ : Subgroup G) : Set G) = Set.univ :=
  rfl


@[to_additive (attr := simp)]
theorem coe_bot : ((⊥ : Subgroup G) : Set G) = {1} :=
  rfl


@[to_additive]
instance : Unique (⊥ : Subgroup G) :=
  ⟨⟨1⟩, fun g => Subtype.ext g.2⟩


@[to_additive (attr := simp)]
theorem top_toSubmonoid : (⊤ : Subgroup G).toSubmonoid = ⊤ :=
  rfl


@[to_additive (attr := simp)]
theorem bot_toSubmonoid : (⊥ : Subgroup G).toSubmonoid = ⊥ :=
  rfl


@[to_additive]
theorem eq_bot_iff_forall : H = ⊥ ↔ ∀ x ∈ H, x = (1 : G) :=
  toSubmonoid_injective.eq_iff.symm.trans <| Submonoid.eq_bot_iff_forall _


@[to_additive]
theorem eq_bot_of_subsingleton [Subsingleton H] : H = ⊥ := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : Subsingleton (Subtype fun x => Membership.mem H x)
    ⊢ Eq H Bot.bot
  -/
  rw [Subgroup.eq_bot_iff_forall]
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : Subsingleton (Subtype fun x => Membership.mem H x)
    ⊢ ∀ (x : G), Membership.mem H x → Eq x 1
  -/
  intro y hy
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : Subsingleton (Subtype fun x => Membership.mem H x)
    y : G
    hy : Membership.mem H y
    ⊢ Eq y 1
  -/
  rw [← Subgroup.coe_mk H y hy, Subsingleton.elim (⟨y, hy⟩ : H) 1, Subgroup.coe_one]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp, norm_cast)]
theorem coe_eq_univ {H : Subgroup G} : (H : Set G) = Set.univ ↔ H = ⊤ :=
                              /-
                                G : Type u_1
                                inst✝ : Group G
                                H : Subgroup G
                                ⊢ Iff (Eq ↑H ↑Top.top) (Eq (↑H) Set.univ)
                              -/
  (SetLike.ext'_iff.trans (by rfl)).symm
                              /-
                                🎉 no goals
                              -/


@[to_additive]
theorem coe_eq_singleton {H : Subgroup G} : (∃ g : G, (H : Set G) = {g}) ↔ H = ⊥ :=
  ⟨fun ⟨g, hg⟩ =>
    haveI : Subsingleton (H : Set G) := by
      /-
        G : Type u_1
        inst✝ : Group G
        H : Subgroup G
        x✝ : Exists fun g => Eq (↑H) (Singleton.singleton g)
        g : G
        hg : Eq (↑H) (Singleton.singleton g)
        ⊢ Subsingleton ↑↑H
      -/
      rw [hg]
      /-
        G : Type u_1
        inst✝ : Group G
        H : Subgroup G
        x✝ : Exists fun g => Eq (↑H) (Singleton.singleton g)
        g : G
        hg : Eq (↑H) (Singleton.singleton g)
        ⊢ Subsingleton ↑(Singleton.singleton g)
      -/
      infer_instance
      /-
        🎉 no goals
      -/
    H.eq_bot_of_subsingleton,
    fun h => ⟨1, SetLike.ext'_iff.mp h⟩⟩


@[to_additive]
theorem nontrivial_iff_exists_ne_one (H : Subgroup G) : Nontrivial H ↔ ∃ x ∈ H, x ≠ (1 : G) := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Iff (Nontrivial (Subtype fun x => Membership.mem H x)) (Exists fun x => And  …
  -/
  rw [Subtype.nontrivial_iff_exists_ne (fun x => x ∈ H) (1 : H)]
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Iff (Exists fun y => Exists fun x => Ne y ↑1) (Exists fun x => And (Membersh …
  -/
  simp
  /-
    🎉 no goals
  -/


@[to_additive]
theorem exists_ne_one_of_nontrivial (H : Subgroup G) [Nontrivial H] :
    ∃ x ∈ H, x ≠ 1 := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    inst✝ : Nontrivial (Subtype fun x => Membership.mem H x)
    ⊢ Exists fun x => And (Membership.mem H x) (Ne x 1)
  -/
  rwa [← Subgroup.nontrivial_iff_exists_ne_one]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem nontrivial_iff_ne_bot (H : Subgroup G) : Nontrivial H ↔ H ≠ ⊥ := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Iff (Nontrivial (Subtype fun x => Membership.mem H x)) (Ne H Bot.bot)
  -/
  rw [nontrivial_iff_exists_ne_one, ne_eq, eq_bot_iff_forall]
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Iff (Exists fun x => And (Membership.mem H x) (Ne x 1)) (Not (∀ (x : G), Mem …
  -/
  simp only [ne_eq, not_forall, exists_prop]
  /-
    🎉 no goals
  -/


/-- A subgroup is either the trivial subgroup or nontrivial. -/
@[to_additive "A subgroup is either the trivial subgroup or nontrivial."]
theorem bot_or_nontrivial (H : Subgroup G) : H = ⊥ ∨ Nontrivial H := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Or (Eq H Bot.bot) (Nontrivial (Subtype fun x => Membership.mem H x))
  -/
  have := nontrivial_iff_ne_bot H
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    this : Iff (Nontrivial (Subtype fun x => Membership.mem H x)) (Ne H Bot.bot)
    ⊢ Or (Eq H Bot.bot) (Nontrivial (Subtype fun x => Membership.mem H x))
  -/
  tauto
  /-
    🎉 no goals
  -/


/-- A subgroup is either the trivial subgroup or contains a non-identity element. -/
@[to_additive "A subgroup is either the trivial subgroup or contains a nonzero element."]
theorem bot_or_exists_ne_one (H : Subgroup G) : H = ⊥ ∨ ∃ x ∈ H, x ≠ (1 : G) := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Or (Eq H Bot.bot) (Exists fun x => And (Membership.mem H x) (Ne x 1))
  -/
  convert H.bot_or_nontrivial
  /-
    case h.e'_2.a
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Iff (Exists fun x => And (Membership.mem H x) (Ne x 1)) (Nontrivial (Subtype …
  -/
  rw [nontrivial_iff_exists_ne_one]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma ne_bot_iff_exists_ne_one {H : Subgroup G} : H ≠ ⊥ ↔ ∃ a : ↥H, a ≠ 1 := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Iff (Ne H Bot.bot) (Exists fun a => Ne a 1)
  -/
  rw [← nontrivial_iff_ne_bot, nontrivial_iff_exists_ne_one]
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    ⊢ Iff (Exists fun x => And (Membership.mem H x) (Ne x 1)) (Exists fun a => Ne  …
  -/
  simp only [ne_eq, Subtype.exists, mk_eq_one, exists_prop]
  /-
    🎉 no goals
  -/


/-- The inf of two subgroups is their intersection. -/
@[to_additive "The inf of two `AddSubgroup`s is their intersection."]
instance : Min (Subgroup G) :=
  ⟨fun H₁ H₂ =>
    { H₁.toSubmonoid ⊓ H₂.toSubmonoid with
      inv_mem' := fun ⟨hx, hx'⟩ => ⟨H₁.inv_mem hx, H₂.inv_mem hx'⟩ }⟩


@[to_additive (attr := simp)]
theorem coe_inf (p p' : Subgroup G) : ((p ⊓ p' : Subgroup G) : Set G) = (p : Set G) ∩ p' :=
  rfl


@[to_additive (attr := simp)]
theorem mem_inf {p p' : Subgroup G} {x : G} : x ∈ p ⊓ p' ↔ x ∈ p ∧ x ∈ p' :=
  Iff.rfl


@[to_additive]
instance : InfSet (Subgroup G) :=
  ⟨fun s =>
                                                               /-
                                                                 G : Type u_1
                                                                 inst✝ : Group G
                                                                 H K : Subgroup G
                                                                 s : Set (Subgroup G)
                                                                 ⊢ Eq (Set.iInter fun S => Set.iInter fun h => ↑S) ↑(iInf fun S => iInf fun h = …
                                                               -/
    { (⨅ S ∈ s, Subgroup.toSubmonoid S).copy (⋂ S ∈ s, ↑S) (by simp) with
                                                               /-
                                                                 🎉 no goals
                                                               -/
      inv_mem' := fun {x} hx =>
                                                 /-
                                                   G : Type u_1
                                                   inst✝ : Group G
                                                   H K : Subgroup G
                                                   s : Set (Subgroup G)
                                                   x : G
                                                   hx : Membership.mem __src✝.carrier x
                                                   i : Subgroup G
                                                   h : Membership.mem s i
                                                   ⊢ Membership.mem i x
                                                 -/
        Set.mem_biInter fun i h => i.inv_mem (by apply Set.mem_iInter₂.1 hx i h) }⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


@[to_additive (attr := simp, norm_cast)]
theorem coe_sInf (H : Set (Subgroup G)) : ((sInf H : Subgroup G) : Set G) = ⋂ s ∈ H, ↑s :=
  rfl


@[to_additive (attr := simp)]
theorem mem_sInf {S : Set (Subgroup G)} {x : G} : x ∈ sInf S ↔ ∀ p ∈ S, x ∈ p :=
  Set.mem_iInter₂


@[to_additive]
theorem mem_iInf {ι : Sort*} {S : ι → Subgroup G} {x : G} : (x ∈ ⨅ i, S i) ↔ ∀ i, x ∈ S i := by
  /-
    G : Type u_1
    inst✝ : Group G
    ι : Sort u_2
    S : ι → Subgroup G
    x : G
    ⊢ Iff (Membership.mem (iInf fun i => S i) x) (∀ (i : ι), Membership.mem (S i) x)
  -/
  simp only [iInf, mem_sInf, Set.forall_mem_range]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp, norm_cast)]
theorem coe_iInf {ι : Sort*} {S : ι → Subgroup G} : (↑(⨅ i, S i) : Set G) = ⋂ i, S i := by
  /-
    G : Type u_1
    inst✝ : Group G
    ι : Sort u_2
    S : ι → Subgroup G
    ⊢ Eq (↑(iInf fun i => S i)) (Set.iInter fun i => ↑(S i))
  -/
  simp only [iInf, coe_sInf, Set.biInter_range]
  /-
    🎉 no goals
  -/


/-- Subgroups of a group form a complete lattice. -/
@[to_additive "The `AddSubgroup`s of an `AddGroup` form a complete lattice."]
instance : CompleteLattice (Subgroup G) :=
  { completeLatticeOfInf (Subgroup G) fun _s =>
      IsGLB.of_image SetLike.coe_subset_coe isGLB_biInf with
    bot := ⊥
    bot_le := fun S _x hx => (mem_bot.1 hx).symm ▸ S.one_mem
    top := ⊤
    le_top := fun _S x _hx => mem_top x
    inf := (· ⊓ ·)
    le_inf := fun _a _b _c ha hb _x hx => ⟨ha hx, hb hx⟩
    inf_le_left := fun _a _b _x => And.left
    inf_le_right := fun _a _b _x => And.right }


@[to_additive]
theorem mem_sup_left {S T : Subgroup G} : ∀ {x : G}, x ∈ S → x ∈ S ⊔ T :=
  have : S ≤ S ⊔ T := le_sup_left; fun h ↦ this h


@[to_additive]
theorem mem_sup_right {S T : Subgroup G} : ∀ {x : G}, x ∈ T → x ∈ S ⊔ T :=
  have : T ≤ S ⊔ T := le_sup_right; fun h ↦ this h


@[to_additive]
theorem mul_mem_sup {S T : Subgroup G} {x y : G} (hx : x ∈ S) (hy : y ∈ T) : x * y ∈ S ⊔ T :=
  (S ⊔ T).mul_mem (mem_sup_left hx) (mem_sup_right hy)


@[to_additive]
theorem mem_iSup_of_mem {ι : Sort*} {S : ι → Subgroup G} (i : ι) :
    ∀ {x : G}, x ∈ S i → x ∈ iSup S :=
  have : S i ≤ iSup S := le_iSup _ _; fun h ↦ this h


@[to_additive]
theorem mem_sSup_of_mem {S : Set (Subgroup G)} {s : Subgroup G} (hs : s ∈ S) :
    ∀ {x : G}, x ∈ s → x ∈ sSup S :=
  have : s ≤ sSup S := le_sSup hs; fun h ↦ this h


@[to_additive (attr := simp)]
theorem subsingleton_iff : Subsingleton (Subgroup G) ↔ Subsingleton G :=
  ⟨fun _ =>
    ⟨fun x y =>
      have : ∀ i : G, i = 1 := fun i =>
        mem_bot.mp <| Subsingleton.elim (⊤ : Subgroup G) ⊥ ▸ mem_top i
      (this x).trans (this y).symm⟩,
                                                                          /-
                                                                            G : Type u_1
                                                                            inst✝ : Group G
                                                                            x✝ : Subsingleton G
                                                                            x y : Subgroup G
                                                                            i : G
                                                                            ⊢ Iff (Membership.mem x 1) (Membership.mem y 1)
                                                                          -/
    fun _ => ⟨fun x y => Subgroup.ext fun i => Subsingleton.elim 1 i ▸ by simp [Subgroup.one_mem]⟩⟩
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[to_additive (attr := simp)]
theorem nontrivial_iff : Nontrivial (Subgroup G) ↔ Nontrivial G :=
  not_iff_not.mp
    ((not_nontrivial_iff_subsingleton.trans subsingleton_iff).trans
      not_nontrivial_iff_subsingleton.symm)


@[to_additive]
instance [Subsingleton G] : Unique (Subgroup G) :=
  ⟨⟨⊥⟩, fun a => @Subsingleton.elim _ (subsingleton_iff.mpr ‹_›) a _⟩


@[to_additive]
instance [Nontrivial G] : Nontrivial (Subgroup G) :=
  nontrivial_iff.mpr ‹_›


@[to_additive]
theorem eq_top_iff' : H = ⊤ ↔ ∀ x : G, x ∈ H :=
  eq_top_iff.trans ⟨fun h m => h <| mem_top m, fun h m _ => h m⟩


/-- The `Subgroup` generated by a set. -/
@[to_additive "The `AddSubgroup` generated by a set"]
def closure (k : Set G) : Subgroup G :=
  sInf { K | k ⊆ K }


@[to_additive]
theorem mem_closure {x : G} : x ∈ closure k ↔ ∀ K : Subgroup G, k ⊆ K → x ∈ K :=
  mem_sInf


/-- The subgroup generated by a set includes the set. -/
@[to_additive (attr := simp, aesop safe 20 apply (rule_sets := [SetLike]))
  "The `AddSubgroup` generated by a set includes the set."]
theorem subset_closure : k ⊆ closure k := fun _ hx => mem_closure.2 fun _ hK => hK hx


@[to_additive]
theorem not_mem_of_not_mem_closure {P : G} (hP : P ∉ closure k) : P ∉ k := fun h =>
  hP (subset_closure h)


/-- A subgroup `K` includes `closure k` if and only if it includes `k`. -/
@[to_additive (attr := simp)
  "An additive subgroup `K` includes `closure k` if and only if it includes `k`"]
theorem closure_le : closure k ≤ K ↔ k ⊆ K :=
  ⟨Subset.trans subset_closure, fun h => sInf_le h⟩


@[to_additive]
theorem closure_eq_of_le (h₁ : k ⊆ K) (h₂ : K ≤ closure k) : closure k = K :=
  le_antisymm ((closure_le <| K).2 h₁) h₂


/-- An induction principle for closure membership. If `p` holds for `1` and all elements of `k`, and
is preserved under multiplication and inverse, then `p` holds for all elements of the closure
of `k`.

See also `Subgroup.closure_induction_left` and `Subgroup.closure_induction_right` for versions that
only require showing `p` is preserved by multiplication by elements in `k`. -/
@[to_additive (attr := elab_as_elim)
      "An induction principle for additive closure membership. If `p`
      holds for `0` and all elements of `k`, and is preserved under addition and inverses, then `p`
      holds for all elements of the additive closure of `k`.

      See also `AddSubgroup.closure_induction_left` and `AddSubgroup.closure_induction_left` for
      versions that only require showing `p` is preserved by addition by elements in `k`."]
theorem closure_induction {p : (g : G) → g ∈ closure k → Prop}
    (mem : ∀ x (hx : x ∈ k), p x (subset_closure hx)) (one : p 1 (one_mem _))
    (mul : ∀ x y hx hy, p x hx → p y hy → p (x * y) (mul_mem hx hy))
    (inv : ∀ x hx, p x hx → p x⁻¹ (inv_mem hx)) {x} (hx : x ∈ closure k) : p x hx :=
  let K : Subgroup G :=
    { carrier := { x | ∃ hx, p x hx }
      mul_mem' := fun ⟨_, ha⟩ ⟨_, hb⟩ ↦ ⟨_, mul _ _ _ _ ha hb⟩
      one_mem' := ⟨_, one⟩
      inv_mem' := fun ⟨_, hb⟩ ↦ ⟨_, inv _ _ hb⟩ }
  closure_le (K := K) |>.mpr (fun y hy ↦ ⟨subset_closure hy, mem y hy⟩) hx |>.elim fun _ ↦ id


@[deprecated closure_induction (since := "2024-10-10")]
alias closure_induction' := closure_induction


/-- An induction principle for closure membership for predicates with two arguments. -/
@[to_additive (attr := elab_as_elim)
      "An induction principle for additive closure membership, for
      predicates with two arguments."]
theorem closure_induction₂ {p : (x y : G) → x ∈ closure k → y ∈ closure k → Prop}
    (mem : ∀ (x) (y) (hx : x ∈ k) (hy : y ∈ k), p x y (subset_closure hx) (subset_closure hy))
    (one_left : ∀ x hx, p 1 x (one_mem _) hx) (one_right : ∀ x hx, p x 1 hx (one_mem _))
    (mul_left : ∀ x y z hx hy hz, p x z hx hz → p y z hy hz → p (x * y) z (mul_mem hx hy) hz)
    (mul_right : ∀ y z x hy hz hx, p x y hx hy → p x z hx hz → p x (y * z) hx (mul_mem hy hz))
    (inv_left : ∀ x y hx hy, p x y hx hy → p x⁻¹ y (inv_mem hx) hy)
    (inv_right : ∀ x y hx hy, p x y hx hy → p x y⁻¹ hx (inv_mem hy))
    {x y : G} (hx : x ∈ closure k) (hy : y ∈ closure k) : p x y hx hy := by
  induction hy using closure_induction with
  | mem z hz => induction hx using closure_induction with
    | mem _ h => exact mem _ _ h hz
    | one => exact one_left _ (subset_closure hz)
    | mul _ _ _ _ h₁ h₂ => exact mul_left _ _ _ _ _ _ h₁ h₂
    | inv _ _ h => exact inv_left _ _ _ _ h
  | one => exact one_right x hx
  | mul _ _ _ _ h₁ h₂ => exact mul_right _ _ _ _ _ hx h₁ h₂
  | inv _ _ h => exact inv_right _ _ _ _ h


@[to_additive (attr := simp)]
theorem closure_closure_coe_preimage {k : Set G} : closure (((↑) : closure k → G) ⁻¹' k) = ⊤ :=
  eq_top_iff.2 fun x _ ↦ Subtype.recOn x fun _ hx' ↦
    closure_induction (fun _ h ↦ subset_closure h) (one_mem _) (fun _ _ _ _ ↦ mul_mem)
      (fun _ _ ↦ inv_mem) hx'


/-- `closure` forms a Galois insertion with the coercion to set. -/
@[to_additive "`closure` forms a Galois insertion with the coercion to set."]
protected def gi : GaloisInsertion (@closure G _) (↑) where
  choice s _ := closure s
  gc s t := @closure_le _ _ t s
  le_l_u _s := subset_closure
  choice_eq _s _h := rfl


/-- Subgroup closure of a set is monotone in its argument: if `h ⊆ k`,
then `closure h ≤ closure k`. -/
@[to_additive (attr := gcongr)
      "Additive subgroup closure of a set is monotone in its argument: if `h ⊆ k`,
      then `closure h ≤ closure k`"]
theorem closure_mono ⦃h k : Set G⦄ (h' : h ⊆ k) : closure h ≤ closure k :=
  (Subgroup.gi G).gc.monotone_l h'


/-- Closure of a subgroup `K` equals `K`. -/
@[to_additive (attr := simp) "Additive closure of an additive subgroup `K` equals `K`"]
theorem closure_eq : closure (K : Set G) = K :=
  (Subgroup.gi G).l_u_eq K


@[to_additive (attr := simp)]
theorem closure_empty : closure (∅ : Set G) = ⊥ :=
  (Subgroup.gi G).gc.l_bot


@[to_additive (attr := simp)]
theorem closure_univ : closure (univ : Set G) = ⊤ :=
  @coe_top G _ ▸ closure_eq ⊤


@[to_additive]
theorem closure_union (s t : Set G) : closure (s ∪ t) = closure s ⊔ closure t :=
  (Subgroup.gi G).gc.l_sup


@[to_additive]
theorem sup_eq_closure (H H' : Subgroup G) : H ⊔ H' = closure ((H : Set G) ∪ (H' : Set G)) := by
  /-
    G : Type u_1
    inst✝ : Group G
    H H' : Subgroup G
    ⊢ Eq (Max.max H H') (Subgroup.closure (Union.union ↑H ↑H'))
  -/
  simp_rw [closure_union, closure_eq]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem closure_iUnion {ι} (s : ι → Set G) : closure (⋃ i, s i) = ⨆ i, closure (s i) :=
  (Subgroup.gi G).gc.l_iSup


@[to_additive (attr := simp)]
theorem closure_eq_bot_iff : closure k = ⊥ ↔ k ⊆ {1} := le_bot_iff.symm.trans <| closure_le _


@[to_additive]
theorem iSup_eq_closure {ι : Sort*} (p : ι → Subgroup G) :
                                                  /-
                                                    G : Type u_1
                                                    inst✝ : Group G
                                                    ι : Sort u_2
                                                    p : ι → Subgroup G
                                                    ⊢ Eq (iSup fun i => p i) (Subgroup.closure (Set.iUnion fun i => ↑(p i)))
                                                  -/
    ⨆ i, p i = closure (⋃ i, (p i : Set G)) := by simp_rw [closure_iUnion, closure_eq]
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- The subgroup generated by an element of a group equals the set of integer number powers of
    the element. -/
@[to_additive
      "The `AddSubgroup` generated by an element of an `AddGroup` equals the set of
      natural number multiples of the element."]
theorem mem_closure_singleton {x y : G} : y ∈ closure ({x} : Set G) ↔ ∃ n : ℤ, x ^ n = y := by
  refine
    ⟨fun hy => closure_induction ?_ ?_ ?_ ?_ hy, fun ⟨n, hn⟩ =>
      hn ▸ zpow_mem (subset_closure <| mem_singleton x) n⟩
    /-
      case refine_1
      G : Type u_1
      inst✝ : Group G
      x y : G
      hy : Membership.mem (Subgroup.closure (Singleton.singleton x)) y
      ⊢ ∀ (x_1 : G), Membership.mem (Singleton.singleton x) x_1 → Exists fun n => Eq …
    -/
  · intro y hy
    /-
      case refine_1
      G : Type u_1
      inst✝ : Group G
      x y✝ : G
      hy✝ : Membership.mem (Subgroup.closure (Singleton.singleton x)) y✝
      y : G
      hy : Membership.mem (Singleton.singleton x) y
      ⊢ Exists fun n => Eq (HPow.hPow x n) y
    -/
    rw [eq_of_mem_singleton hy]
    /-
      case refine_1
      G : Type u_1
      inst✝ : Group G
      x y✝ : G
      hy✝ : Membership.mem (Subgroup.closure (Singleton.singleton x)) y✝
      y : G
      hy : Membership.mem (Singleton.singleton x) y
      ⊢ Exists fun n => Eq (HPow.hPow x n) x
    -/
    exact ⟨1, zpow_one x⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_1
      inst✝ : Group G
      x y : G
      hy : Membership.mem (Subgroup.closure (Singleton.singleton x)) y
      ⊢ Exists fun n => Eq (HPow.hPow x n) 1
    -/
  · exact ⟨0, zpow_zero x⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      G : Type u_1
      inst✝ : Group G
      x y : G
      hy : Membership.mem (Subgroup.closure (Singleton.singleton x)) y
      ⊢ ∀ (x_1 y : G), Membership.mem (Subgroup.closure (Singleton.singleton x)) x_1 …
    -/
  · rintro _ _ _ _ ⟨n, rfl⟩ ⟨m, rfl⟩
    /-
      case refine_3.intro.intro
      G : Type u_1
      inst✝ : Group G
      x y : G
      hy : Membership.mem (Subgroup.closure (Singleton.singleton x)) y
      n : Int
      hx✝ : Membership.mem (Subgroup.closure (Singleton.singleton x)) (HPow.hPow x n)
      m : Int
      hy✝ : Membership.mem (Subgroup.closure (Singleton.singleton x)) (HPow.hPow x m)
      ⊢ Exists fun n_1 => Eq (HPow.hPow x n_1) (HMul.hMul (HPow.hPow x n) (HPow.hPow …
    -/
    exact ⟨n + m, zpow_add x n m⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_4
    G : Type u_1
    inst✝ : Group G
    x y : G
    hy : Membership.mem (Subgroup.closure (Singleton.singleton x)) y
    ⊢ ∀ (x_1 : G), Membership.mem (Subgroup.closure (Singleton.singleton x)) x_1 → …
  -/
  rintro _ _ ⟨n, rfl⟩
  /-
    case refine_4.intro
    G : Type u_1
    inst✝ : Group G
    x y : G
    hy : Membership.mem (Subgroup.closure (Singleton.singleton x)) y
    n : Int
    hx✝ : Membership.mem (Subgroup.closure (Singleton.singleton x)) (HPow.hPow x n)
    ⊢ Exists fun n_1 => Eq (HPow.hPow x n_1) (Inv.inv (HPow.hPow x n))
  -/
  exact ⟨-n, zpow_neg x n⟩
  /-
    🎉 no goals
  -/


@[to_additive]
theorem closure_singleton_one : closure ({1} : Set G) = ⊥ := by
  /-
    G : Type u_1
    inst✝ : Group G
    ⊢ Eq (Subgroup.closure (Singleton.singleton 1)) Bot.bot
  -/
  simp [eq_bot_iff_forall, mem_closure_singleton]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma mem_closure_singleton_self (x : G) : x ∈ closure ({x} : Set G) := by
  /-
    G : Type u_1
    inst✝ : Group G
    x : G
    ⊢ Membership.mem (Subgroup.closure (Singleton.singleton x)) x
  -/
  simpa [-subset_closure] using subset_closure (k := {x})
  /-
    🎉 no goals
  -/


@[to_additive]
theorem le_closure_toSubmonoid (S : Set G) : Submonoid.closure S ≤ (closure S).toSubmonoid :=
  Submonoid.closure_le.2 subset_closure


@[to_additive]
theorem closure_eq_top_of_mclosure_eq_top {S : Set G} (h : Submonoid.closure S = ⊤) :
    closure S = ⊤ :=
  (eq_top_iff' _).2 fun _ => le_closure_toSubmonoid _ <| h.symm ▸ trivial


@[to_additive]
theorem mem_iSup_of_directed {ι} [hι : Nonempty ι] {K : ι → Subgroup G} (hK : Directed (· ≤ ·) K)
    {x : G} : x ∈ (iSup K : Subgroup G) ↔ ∃ i, x ∈ K i := by
  /-
    G : Type u_1
    inst✝ : Group G
    ι : Sort u_2
    hι : Nonempty ι
    K : ι → Subgroup G
    hK : Directed (fun x1 x2 => LE.le x1 x2) K
    x : G
    ⊢ Iff (Membership.mem (iSup K) x) (Exists fun i => Membership.mem (K i) x)
  -/
  refine ⟨?_, fun ⟨i, hi⟩ ↦ le_iSup K i hi⟩
  suffices x ∈ closure (⋃ i, (K i : Set G)) → ∃ i, x ∈ K i by
    simpa only [closure_iUnion, closure_eq (K _)] using this
  /-
    G : Type u_1
    inst✝ : Group G
    ι : Sort u_2
    hι : Nonempty ι
    K : ι → Subgroup G
    hK : Directed (fun x1 x2 => LE.le x1 x2) K
    x : G
    ⊢ Membership.mem (Subgroup.closure (Set.iUnion fun i => ↑(K i))) x → Exists fu …
  -/
  refine fun hx ↦ closure_induction (fun _ ↦ mem_iUnion.1) ?_ ?_ ?_ hx
    /-
      case refine_1
      G : Type u_1
      inst✝ : Group G
      ι : Sort u_2
      hι : Nonempty ι
      K : ι → Subgroup G
      hK : Directed (fun x1 x2 => LE.le x1 x2) K
      x : G
      hx : Membership.mem (Subgroup.closure (Set.iUnion fun i => ↑(K i))) x
      ⊢ Exists fun i => Membership.mem (K i) 1
    -/
  · exact hι.elim fun i ↦ ⟨i, (K i).one_mem⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_1
      inst✝ : Group G
      ι : Sort u_2
      hι : Nonempty ι
      K : ι → Subgroup G
      hK : Directed (fun x1 x2 => LE.le x1 x2) K
      x : G
      hx : Membership.mem (Subgroup.closure (Set.iUnion fun i => ↑(K i))) x
      ⊢ ∀ (x y : G), Membership.mem (Subgroup.closure (Set.iUnion fun i => ↑(K i)))  …
    -/
  · rintro x y _ _ ⟨i, hi⟩ ⟨j, hj⟩
    /-
      case refine_2.intro.intro
      G : Type u_1
      inst✝ : Group G
      ι : Sort u_2
      hι : Nonempty ι
      K : ι → Subgroup G
      hK : Directed (fun x1 x2 => LE.le x1 x2) K
      x✝ : G
      hx : Membership.mem (Subgroup.closure (Set.iUnion fun i => ↑(K i))) x✝
      x y : G
      hx✝ : Membership.mem (Subgroup.closure (Set.iUnion fun i => ↑(K i))) x
      hy✝ : Membership.mem (Subgroup.closure (Set.iUnion fun i => ↑(K i))) y
      i : ι
      hi : Membership.mem (K i) x
      j : ι
      hj : Membership.mem (K j) y
      ⊢ Exists fun i => Membership.mem (K i) (HMul.hMul x y)
    -/
    rcases hK i j with ⟨k, hki, hkj⟩
    /-
      case refine_2.intro.intro.intro.intro
      G : Type u_1
      inst✝ : Group G
      ι : Sort u_2
      hι : Nonempty ι
      K : ι → Subgroup G
      hK : Directed (fun x1 x2 => LE.le x1 x2) K
      x✝ : G
      hx : Membership.mem (Subgroup.closure (Set.iUnion fun i => ↑(K i))) x✝
      x y : G
      hx✝ : Membership.mem (Subgroup.closure (Set.iUnion fun i => ↑(K i))) x
      hy✝ : Membership.mem (Subgroup.closure (Set.iUnion fun i => ↑(K i))) y
      i : ι
      hi : Membership.mem (K i) x
      j : ι
      hj : Membership.mem (K j) y
      k : ι
      hki : LE.le (K i) (K k)
      hkj : LE.le (K j) (K k)
      ⊢ Exists fun i => Membership.mem (K i) (HMul.hMul x y)
    -/
    exact ⟨k, mul_mem (hki hi) (hkj hj)⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      G : Type u_1
      inst✝ : Group G
      ι : Sort u_2
      hι : Nonempty ι
      K : ι → Subgroup G
      hK : Directed (fun x1 x2 => LE.le x1 x2) K
      x : G
      hx : Membership.mem (Subgroup.closure (Set.iUnion fun i => ↑(K i))) x
      ⊢ ∀ (x : G), Membership.mem (Subgroup.closure (Set.iUnion fun i => ↑(K i))) x  …
    -/
  · rintro _ _ ⟨i, hi⟩
    /-
      case refine_3.intro
      G : Type u_1
      inst✝ : Group G
      ι : Sort u_2
      hι : Nonempty ι
      K : ι → Subgroup G
      hK : Directed (fun x1 x2 => LE.le x1 x2) K
      x : G
      hx : Membership.mem (Subgroup.closure (Set.iUnion fun i => ↑(K i))) x
      x✝ : G
      hx✝ : Membership.mem (Subgroup.closure (Set.iUnion fun i => ↑(K i))) x✝
      i : ι
      hi : Membership.mem (K i) x✝
      ⊢ Exists fun i => Membership.mem (K i) (Inv.inv x✝)
    -/
    exact ⟨i, inv_mem hi⟩
    /-
      🎉 no goals
    -/


@[to_additive]
theorem coe_iSup_of_directed {ι} [Nonempty ι] {S : ι → Subgroup G} (hS : Directed (· ≤ ·) S) :
    ((⨆ i, S i : Subgroup G) : Set G) = ⋃ i, S i :=
                     /-
                       G : Type u_1
                       inst✝¹ : Group G
                       ι : Sort u_2
                       inst✝ : Nonempty ι
                       S : ι → Subgroup G
                       hS : Directed (fun x1 x2 => LE.le x1 x2) S
                       x : G
                       ⊢ Iff (Membership.mem (↑(iSup fun i => S i)) x) (Membership.mem (Set.iUnion fu …
                     -/
  Set.ext fun x ↦ by simp [mem_iSup_of_directed hS]
                     /-
                       🎉 no goals
                     -/


@[to_additive]
theorem mem_sSup_of_directedOn {K : Set (Subgroup G)} (Kne : K.Nonempty) (hK : DirectedOn (· ≤ ·) K)
    {x : G} : x ∈ sSup K ↔ ∃ s ∈ K, x ∈ s := by
  /-
    G : Type u_1
    inst✝ : Group G
    K : Set (Subgroup G)
    Kne : K.Nonempty
    hK : DirectedOn (fun x1 x2 => LE.le x1 x2) K
    x : G
    ⊢ Iff (Membership.mem (SupSet.sSup K) x) (Exists fun s => And (Membership.mem  …
  -/
  haveI : Nonempty K := Kne.to_subtype
  simp only [sSup_eq_iSup', mem_iSup_of_directed hK.directed_val, SetCoe.exists, Subtype.coe_mk,
    exists_prop]


@[to_additive]
theorem mem_sup : x ∈ s ⊔ t ↔ ∃ y ∈ s, ∃ z ∈ t, y * z = x :=
  ⟨fun h => by
    /-
      C : Type u_2
      inst✝ : CommGroup C
      s t : Subgroup C
      x : C
      h : Membership.mem (Max.max s t) x
      ⊢ Exists fun y => And (Membership.mem s y) (Exists fun z => And (Membership.me …
    -/
    rw [sup_eq_closure] at h
    /-
      C : Type u_2
      inst✝ : CommGroup C
      s t : Subgroup C
      x : C
      h : Membership.mem (Subgroup.closure (Union.union ↑s ↑t)) x
      ⊢ Exists fun y => And (Membership.mem s y) (Exists fun z => And (Membership.me …
    -/
    refine Subgroup.closure_induction ?_ ?_ ?_ ?_ h
      /-
        case refine_1
        C : Type u_2
        inst✝ : CommGroup C
        s t : Subgroup C
        x : C
        h : Membership.mem (Subgroup.closure (Union.union ↑s ↑t)) x
        ⊢ ∀ (x : C), Membership.mem (Union.union ↑s ↑t) x → Exists fun y => And (Membe …
      -/
    · rintro y (h | h)
        /-
          case refine_1.inl
          C : Type u_2
          inst✝ : CommGroup C
          s t : Subgroup C
          x : C
          h✝ : Membership.mem (Subgroup.closure (Union.union ↑s ↑t)) x
          y : C
          h : Membership.mem (↑s) y
          ⊢ Exists fun y_1 => And (Membership.mem s y_1) (Exists fun z => And (Membershi …
        -/
      · exact ⟨y, h, 1, t.one_mem, by simp⟩
        /-
          🎉 no goals
        -/
        /-
          case refine_1.inr
          C : Type u_2
          inst✝ : CommGroup C
          s t : Subgroup C
          x : C
          h✝ : Membership.mem (Subgroup.closure (Union.union ↑s ↑t)) x
          y : C
          h : Membership.mem (↑t) y
          ⊢ Exists fun y_1 => And (Membership.mem s y_1) (Exists fun z => And (Membershi …
        -/
      · exact ⟨1, s.one_mem, y, h, by simp⟩
        /-
          🎉 no goals
        -/
      /-
        case refine_2
        C : Type u_2
        inst✝ : CommGroup C
        s t : Subgroup C
        x : C
        h : Membership.mem (Subgroup.closure (Union.union ↑s ↑t)) x
        ⊢ Exists fun y => And (Membership.mem s y) (Exists fun z => And (Membership.me …
      -/
    · exact ⟨1, s.one_mem, 1, ⟨t.one_mem, mul_one 1⟩⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        C : Type u_2
        inst✝ : CommGroup C
        s t : Subgroup C
        x : C
        h : Membership.mem (Subgroup.closure (Union.union ↑s ↑t)) x
        ⊢ ∀ (x y : C), Membership.mem (Subgroup.closure (Union.union ↑s ↑t)) x → Membe …
      -/
    · rintro _ _ _ _ ⟨y₁, hy₁, z₁, hz₁, rfl⟩ ⟨y₂, hy₂, z₂, hz₂, rfl⟩
      /-
        case refine_3.intro.intro.intro.intro.intro.intro.intro.intro
        C : Type u_2
        inst✝ : CommGroup C
        s t : Subgroup C
        x : C
        h : Membership.mem (Subgroup.closure (Union.union ↑s ↑t)) x
        y₁ : C
        hy₁ : Membership.mem s y₁
        z₁ : C
        hz₁ : Membership.mem t z₁
        hx✝ : Membership.mem (Subgroup.closure (Union.union ↑s ↑t)) (HMul.hMul y₁ z₁)
        y₂ : C
        hy₂ : Membership.mem s y₂
        z₂ : C
        hz₂ : Membership.mem t z₂
        hy✝ : Membership.mem (Subgroup.closure (Union.union ↑s ↑t)) (HMul.hMul y₂ z₂)
        ⊢ Exists fun y => And (Membership.mem s y) (Exists fun z => And (Membership.me …
      -/
      exact ⟨_, mul_mem hy₁ hy₂, _, mul_mem hz₁ hz₂, by simp [mul_assoc, mul_left_comm]⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_4
        C : Type u_2
        inst✝ : CommGroup C
        s t : Subgroup C
        x : C
        h : Membership.mem (Subgroup.closure (Union.union ↑s ↑t)) x
        ⊢ ∀ (x : C), Membership.mem (Subgroup.closure (Union.union ↑s ↑t)) x → (Exists …
      -/
    · rintro _ _ ⟨y, hy, z, hz, rfl⟩
      /-
        case refine_4.intro.intro.intro.intro
        C : Type u_2
        inst✝ : CommGroup C
        s t : Subgroup C
        x : C
        h : Membership.mem (Subgroup.closure (Union.union ↑s ↑t)) x
        y : C
        hy : Membership.mem s y
        z : C
        hz : Membership.mem t z
        hx✝ : Membership.mem (Subgroup.closure (Union.union ↑s ↑t)) (HMul.hMul y z)
        ⊢ Exists fun y_1 => And (Membership.mem s y_1) (Exists fun z_1 => And (Members …
      -/
      exact ⟨_, inv_mem hy, _, inv_mem hz, mul_comm z y ▸ (mul_inv_rev z y).symm⟩, by
      /-
        🎉 no goals
      -/
    /-
      C : Type u_2
      inst✝ : CommGroup C
      s t : Subgroup C
      x : C
      ⊢ (Exists fun y => And (Membership.mem s y) (Exists fun z => And (Membership.m …
    -/
    rintro ⟨y, hy, z, hz, rfl⟩; exact mul_mem_sup hy hz⟩
                                /-
                                  🎉 no goals
                                -/


@[to_additive]
theorem mem_sup' : x ∈ s ⊔ t ↔ ∃ (y : s) (z : t), (y : C) * z = x :=
                      /-
                        C : Type u_2
                        inst✝ : CommGroup C
                        s t : Subgroup C
                        x : C
                        ⊢ Iff (Exists fun y => And (Membership.mem s y) (Exists fun z => And (Membersh …
                      -/
  mem_sup.trans <| by simp only [SetLike.exists, coe_mk, exists_prop]
                      /-
                        🎉 no goals
                      -/


@[to_additive]
theorem mem_closure_pair {x y z : C} :
    z ∈ closure ({x, y} : Set C) ↔ ∃ m n : ℤ, x ^ m * y ^ n = z := by
  /-
    C : Type u_2
    inst✝ : CommGroup C
    x y z : C
    ⊢ Iff (Membership.mem (Subgroup.closure (Insert.insert x (Singleton.singleton  …
  -/
  rw [← Set.singleton_union, Subgroup.closure_union, mem_sup]
  /-
    C : Type u_2
    inst✝ : CommGroup C
    x y z : C
    ⊢ Iff (Exists fun y_1 => And (Membership.mem (Subgroup.closure (Singleton.sing …
  -/
  simp_rw [mem_closure_singleton, exists_exists_eq_and]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem disjoint_def {H₁ H₂ : Subgroup G} : Disjoint H₁ H₂ ↔ ∀ {x : G}, x ∈ H₁ → x ∈ H₂ → x = 1 :=
                                  /-
                                    G : Type u_1
                                    inst✝ : Group G
                                    H₁ H₂ : Subgroup G
                                    ⊢ Iff (LE.le (Min.min H₁ H₂) Bot.bot) (∀ {x : G}, Membership.mem H₁ x → Member …
                                  -/
  disjoint_iff_inf_le.trans <| by simp only [Disjoint, SetLike.le_def, mem_inf, mem_bot, and_imp]
                                  /-
                                    🎉 no goals
                                  -/


@[to_additive]
theorem disjoint_def' {H₁ H₂ : Subgroup G} :
    Disjoint H₁ H₂ ↔ ∀ {x y : G}, x ∈ H₁ → y ∈ H₂ → x = y → x = 1 :=
  disjoint_def.trans ⟨fun h _x _y hx hy hxy ↦ h hx <| hxy.symm ▸ hy, fun h _x hx hx' ↦ h hx hx' rfl⟩


@[to_additive]
theorem disjoint_iff_mul_eq_one {H₁ H₂ : Subgroup G} :
    Disjoint H₁ H₂ ↔ ∀ {x y : G}, x ∈ H₁ → y ∈ H₂ → x * y = 1 → x = 1 ∧ y = 1 :=
  disjoint_def'.trans
    ⟨fun h x y hx hy hxy =>
      let hx1 : x = 1 := h hx (H₂.inv_mem hy) (eq_inv_iff_mul_eq_one.mpr hxy)
               /-
                 G : Type u_1
                 inst✝ : Group G
                 H₁ H₂ : Subgroup G
                 h : ∀ {x y : G}, Membership.mem H₁ x → Membership.mem H₂ y → Eq x y → Eq x 1
                 x y : G
                 hx : Membership.mem H₁ x
                 hy : Membership.mem H₂ y
                 hxy : Eq (HMul.hMul x y) 1
                 hx1 : Eq x 1 := h hx (Subgroup.inv_mem H₂ hy) (eq_inv_iff_mul_eq_one.mpr hxy)
                 ⊢ Eq y 1
               -/
      ⟨hx1, by simpa [hx1] using hxy⟩,
               /-
                 🎉 no goals
               -/
      fun h _ _ hx hy hxy => (h hx (H₂.inv_mem hy) (mul_inv_eq_one.mpr hxy)).1⟩


@[to_additive]
theorem mul_injective_of_disjoint {H₁ H₂ : Subgroup G} (h : Disjoint H₁ H₂) :
    Function.Injective (fun g => g.1 * g.2 : H₁ × H₂ → G) := by
  /-
    G : Type u_1
    inst✝ : Group G
    H₁ H₂ : Subgroup G
    h : Disjoint H₁ H₂
    ⊢ Function.Injective fun g => HMul.hMul ↑g.1 ↑g.2
  -/
  intro x y hxy
  /-
    G : Type u_1
    inst✝ : Group G
    H₁ H₂ : Subgroup G
    h : Disjoint H₁ H₂
    x y : Prod (Subtype fun x => Membership.mem H₁ x) (Subtype fun x => Membership …
    hxy : Eq ((fun g => HMul.hMul ↑g.1 ↑g.2) x) ((fun g => HMul.hMul ↑g.1 ↑g.2) y)
    ⊢ Eq x y
  -/
  rw [← inv_mul_eq_iff_eq_mul, ← mul_assoc, ← mul_inv_eq_one, mul_assoc] at hxy
  /-
    G : Type u_1
    inst✝ : Group G
    H₁ H₂ : Subgroup G
    h : Disjoint H₁ H₂
    x y : Prod (Subtype fun x => Membership.mem H₁ x) (Subtype fun x => Membership …
    hxy : Eq (HMul.hMul (HMul.hMul (Inv.inv ↑y.1) ↑x.1) (HMul.hMul (↑x.2) (Inv.inv …
    ⊢ Eq x y
  -/
  replace hxy := disjoint_iff_mul_eq_one.mp h (y.1⁻¹ * x.1).prop (x.2 * y.2⁻¹).prop hxy
  rwa [coe_mul, coe_mul, coe_inv, coe_inv, inv_mul_eq_one, mul_inv_eq_one, ← Subtype.ext_iff, ←
    Subtype.ext_iff, eq_comm, ← Prod.ext_iff] at hxy



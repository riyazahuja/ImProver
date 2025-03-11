@[to_additive]
theorem mem_iSup_of_directed {ι} [hι : Nonempty ι] {S : ι → Submonoid M} (hS : Directed (· ≤ ·) S)
    {x : M} : (x ∈ ⨆ i, S i) ↔ ∃ i, x ∈ S i := by
  /-
    M : Type u_1
    inst✝ : MulOneClass M
    ι : Sort u_4
    hι : Nonempty ι
    S : ι → Submonoid M
    hS : Directed (fun x1 x2 => LE.le x1 x2) S
    x : M
    ⊢ Iff (Membership.mem (iSup fun i => S i) x) (Exists fun i => Membership.mem ( …
  -/
  refine ⟨?_, fun ⟨i, hi⟩ ↦ le_iSup S i hi⟩
  suffices x ∈ closure (⋃ i, (S i : Set M)) → ∃ i, x ∈ S i by
    simpa only [closure_iUnion, closure_eq (S _)] using this
  /-
    M : Type u_1
    inst✝ : MulOneClass M
    ι : Sort u_4
    hι : Nonempty ι
    S : ι → Submonoid M
    hS : Directed (fun x1 x2 => LE.le x1 x2) S
    x : M
    ⊢ Membership.mem (Submonoid.closure (Set.iUnion fun i => ↑(S i))) x → Exists f …
  -/
  refine closure_induction (fun _ ↦ mem_iUnion.1) ?_ ?_
    /-
      case refine_1
      M : Type u_1
      inst✝ : MulOneClass M
      ι : Sort u_4
      hι : Nonempty ι
      S : ι → Submonoid M
      hS : Directed (fun x1 x2 => LE.le x1 x2) S
      x : M
      ⊢ Exists fun i => Membership.mem (S i) 1
    -/
  · exact hι.elim fun i ↦ ⟨i, (S i).one_mem⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      M : Type u_1
      inst✝ : MulOneClass M
      ι : Sort u_4
      hι : Nonempty ι
      S : ι → Submonoid M
      hS : Directed (fun x1 x2 => LE.le x1 x2) S
      x : M
      ⊢ ∀ (x y : M), Membership.mem (Submonoid.closure (Set.iUnion fun i => ↑(S i))) …
    -/
  · rintro x y - - ⟨i, hi⟩ ⟨j, hj⟩
    /-
      case refine_2.intro.intro
      M : Type u_1
      inst✝ : MulOneClass M
      ι : Sort u_4
      hι : Nonempty ι
      S : ι → Submonoid M
      hS : Directed (fun x1 x2 => LE.le x1 x2) S
      x✝ x y : M
      i : ι
      hi : Membership.mem (S i) x
      j : ι
      hj : Membership.mem (S j) y
      ⊢ Exists fun i => Membership.mem (S i) (HMul.hMul x y)
    -/
    rcases hS i j with ⟨k, hki, hkj⟩
    /-
      case refine_2.intro.intro.intro.intro
      M : Type u_1
      inst✝ : MulOneClass M
      ι : Sort u_4
      hι : Nonempty ι
      S : ι → Submonoid M
      hS : Directed (fun x1 x2 => LE.le x1 x2) S
      x✝ x y : M
      i : ι
      hi : Membership.mem (S i) x
      j : ι
      hj : Membership.mem (S j) y
      k : ι
      hki : LE.le (S i) (S k)
      hkj : LE.le (S j) (S k)
      ⊢ Exists fun i => Membership.mem (S i) (HMul.hMul x y)
    -/
    exact ⟨k, (S k).mul_mem (hki hi) (hkj hj)⟩
    /-
      🎉 no goals
    -/


@[to_additive]
theorem coe_iSup_of_directed {ι} [Nonempty ι] {S : ι → Submonoid M} (hS : Directed (· ≤ ·) S) :
    ((⨆ i, S i : Submonoid M) : Set M) = ⋃ i, S i :=
                     /-
                       M : Type u_1
                       inst✝¹ : MulOneClass M
                       ι : Sort u_4
                       inst✝ : Nonempty ι
                       S : ι → Submonoid M
                       hS : Directed (fun x1 x2 => LE.le x1 x2) S
                       x : M
                       ⊢ Iff (Membership.mem (↑(iSup fun i => S i)) x) (Membership.mem (Set.iUnion fu …
                     -/
  Set.ext fun x ↦ by simp [mem_iSup_of_directed hS]
                     /-
                       🎉 no goals
                     -/


@[to_additive]
theorem mem_sSup_of_directedOn {S : Set (Submonoid M)} (Sne : S.Nonempty)
    (hS : DirectedOn (· ≤ ·) S) {x : M} : x ∈ sSup S ↔ ∃ s ∈ S, x ∈ s := by
  /-
    M : Type u_1
    inst✝ : MulOneClass M
    S : Set (Submonoid M)
    Sne : S.Nonempty
    hS : DirectedOn (fun x1 x2 => LE.le x1 x2) S
    x : M
    ⊢ Iff (Membership.mem (SupSet.sSup S) x) (Exists fun s => And (Membership.mem  …
  -/
  haveI : Nonempty S := Sne.to_subtype
  /-
    M : Type u_1
    inst✝ : MulOneClass M
    S : Set (Submonoid M)
    Sne : S.Nonempty
    hS : DirectedOn (fun x1 x2 => LE.le x1 x2) S
    x : M
    this : Nonempty ↑S
    ⊢ Iff (Membership.mem (SupSet.sSup S) x) (Exists fun s => And (Membership.mem  …
  -/
  simp [sSup_eq_iSup', mem_iSup_of_directed hS.directed_val, SetCoe.exists, Subtype.coe_mk]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem coe_sSup_of_directedOn {S : Set (Submonoid M)} (Sne : S.Nonempty)
    (hS : DirectedOn (· ≤ ·) S) : (↑(sSup S) : Set M) = ⋃ s ∈ S, ↑s :=
                      /-
                        M : Type u_1
                        inst✝ : MulOneClass M
                        S : Set (Submonoid M)
                        Sne : S.Nonempty
                        hS : DirectedOn (fun x1 x2 => LE.le x1 x2) S
                        x : M
                        ⊢ Iff (Membership.mem (↑(SupSet.sSup S)) x) (Membership.mem (Set.iUnion fun s  …
                      -/
  Set.ext fun x => by simp [mem_sSup_of_directedOn Sne hS]
                      /-
                        🎉 no goals
                      -/


@[to_additive]
theorem mem_sup_left {S T : Submonoid M} : ∀ {x : M}, x ∈ S → x ∈ S ⊔ T := by
  /-
    M : Type u_1
    inst✝ : MulOneClass M
    S T : Submonoid M
    ⊢ ∀ {x : M}, Membership.mem S x → Membership.mem (Max.max S T) x
  -/
  rw [← SetLike.le_def]
  /-
    M : Type u_1
    inst✝ : MulOneClass M
    S T : Submonoid M
    ⊢ LE.le S (Max.max S T)
  -/
  exact le_sup_left
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mem_sup_right {S T : Submonoid M} : ∀ {x : M}, x ∈ T → x ∈ S ⊔ T := by
  /-
    M : Type u_1
    inst✝ : MulOneClass M
    S T : Submonoid M
    ⊢ ∀ {x : M}, Membership.mem T x → Membership.mem (Max.max S T) x
  -/
  rw [← SetLike.le_def]
  /-
    M : Type u_1
    inst✝ : MulOneClass M
    S T : Submonoid M
    ⊢ LE.le T (Max.max S T)
  -/
  exact le_sup_right
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_mem_sup {S T : Submonoid M} {x y : M} (hx : x ∈ S) (hy : y ∈ T) : x * y ∈ S ⊔ T :=
  (S ⊔ T).mul_mem (mem_sup_left hx) (mem_sup_right hy)


@[to_additive]
theorem mem_iSup_of_mem {ι : Sort*} {S : ι → Submonoid M} (i : ι) :
    ∀ {x : M}, x ∈ S i → x ∈ iSup S := by
  /-
    M : Type u_1
    inst✝ : MulOneClass M
    ι : Sort u_4
    S : ι → Submonoid M
    i : ι
    ⊢ ∀ {x : M}, Membership.mem (S i) x → Membership.mem (iSup S) x
  -/
  rw [← SetLike.le_def]
  /-
    M : Type u_1
    inst✝ : MulOneClass M
    ι : Sort u_4
    S : ι → Submonoid M
    i : ι
    ⊢ LE.le (S i) (iSup S)
  -/
  exact le_iSup _ _
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mem_sSup_of_mem {S : Set (Submonoid M)} {s : Submonoid M} (hs : s ∈ S) :
    ∀ {x : M}, x ∈ s → x ∈ sSup S := by
  /-
    M : Type u_1
    inst✝ : MulOneClass M
    S : Set (Submonoid M)
    s : Submonoid M
    hs : Membership.mem S s
    ⊢ ∀ {x : M}, Membership.mem s x → Membership.mem (SupSet.sSup S) x
  -/
  rw [← SetLike.le_def]
  /-
    M : Type u_1
    inst✝ : MulOneClass M
    S : Set (Submonoid M)
    s : Submonoid M
    hs : Membership.mem S s
    ⊢ LE.le s (SupSet.sSup S)
  -/
  exact le_sSup hs
  /-
    🎉 no goals
  -/


/-- An induction principle for elements of `⨆ i, S i`.
If `C` holds for `1` and all elements of `S i` for all `i`, and is preserved under multiplication,
then it holds for all elements of the supremum of `S`. -/
@[to_additive (attr := elab_as_elim)
      " An induction principle for elements of `⨆ i, S i`.
      If `C` holds for `0` and all elements of `S i` for all `i`, and is preserved under addition,
      then it holds for all elements of the supremum of `S`. "]
theorem iSup_induction {ι : Sort*} (S : ι → Submonoid M) {C : M → Prop} {x : M} (hx : x ∈ ⨆ i, S i)
    (mem : ∀ (i), ∀ x ∈ S i, C x) (one : C 1) (mul : ∀ x y, C x → C y → C (x * y)) : C x := by
  /-
    M : Type u_1
    inst✝ : MulOneClass M
    ι : Sort u_4
    S : ι → Submonoid M
    C : M → Prop
    x : M
    hx : Membership.mem (iSup fun i => S i) x
    mem : ∀ (i : ι) (x : M), Membership.mem (S i) x → C x
    one : C 1
    mul : ∀ (x y : M), C x → C y → C (HMul.hMul x y)
    ⊢ C x
  -/
  rw [iSup_eq_closure] at hx
  /-
    M : Type u_1
    inst✝ : MulOneClass M
    ι : Sort u_4
    S : ι → Submonoid M
    C : M → Prop
    x : M
    hx : Membership.mem (Submonoid.closure (Set.iUnion fun i => ↑(S i))) x
    mem : ∀ (i : ι) (x : M), Membership.mem (S i) x → C x
    one : C 1
    mul : ∀ (x y : M), C x → C y → C (HMul.hMul x y)
    ⊢ C x
  -/
  refine closure_induction (fun x hx => ?_) one (fun _ _ _ _ ↦ mul _ _) hx
  /-
    M : Type u_1
    inst✝ : MulOneClass M
    ι : Sort u_4
    S : ι → Submonoid M
    C : M → Prop
    x✝ : M
    hx✝ : Membership.mem (Submonoid.closure (Set.iUnion fun i => ↑(S i))) x✝
    mem : ∀ (i : ι) (x : M), Membership.mem (S i) x → C x
    one : C 1
    mul : ∀ (x y : M), C x → C y → C (HMul.hMul x y)
    x : M
    hx : Membership.mem (Set.iUnion fun i => ↑(S i)) x
    ⊢ C x
  -/
  obtain ⟨i, hi⟩ := Set.mem_iUnion.mp hx
  /-
    case intro
    M : Type u_1
    inst✝ : MulOneClass M
    ι : Sort u_4
    S : ι → Submonoid M
    C : M → Prop
    x✝ : M
    hx✝ : Membership.mem (Submonoid.closure (Set.iUnion fun i => ↑(S i))) x✝
    mem : ∀ (i : ι) (x : M), Membership.mem (S i) x → C x
    one : C 1
    mul : ∀ (x y : M), C x → C y → C (HMul.hMul x y)
    x : M
    hx : Membership.mem (Set.iUnion fun i => ↑(S i)) x
    i : ι
    hi : Membership.mem (↑(S i)) x
    ⊢ C x
  -/
  exact mem _ _ hi
  /-
    🎉 no goals
  -/


/-- A dependent version of `Submonoid.iSup_induction`. -/
@[to_additive (attr := elab_as_elim) "A dependent version of `AddSubmonoid.iSup_induction`. "]
theorem iSup_induction' {ι : Sort*} (S : ι → Submonoid M) {C : ∀ x, (x ∈ ⨆ i, S i) → Prop}
    (mem : ∀ (i), ∀ (x) (hxS : x ∈ S i), C x (mem_iSup_of_mem i hxS)) (one : C 1 (one_mem _))
    (mul : ∀ x y hx hy, C x hx → C y hy → C (x * y) (mul_mem ‹_› ‹_›)) {x : M}
    (hx : x ∈ ⨆ i, S i) : C x hx := by
  /-
    M : Type u_1
    inst✝ : MulOneClass M
    ι : Sort u_4
    S : ι → Submonoid M
    C : (x : M) → Membership.mem (iSup fun i => S i) x → Prop
    mem : ∀ (i : ι) (x : M) (hxS : Membership.mem (S i) x), C x ⋯
    one : C 1 ⋯
    mul : ∀ (x y : M) (hx : Membership.mem (iSup fun i => S i) x) (hy : Membership …
    x : M
    hx : Membership.mem (iSup fun i => S i) x
    ⊢ C x hx
  -/
  refine Exists.elim (?_ : ∃ Hx, C x Hx) fun (hx : x ∈ ⨆ i, S i) (hc : C x hx) => hc
  /-
    M : Type u_1
    inst✝ : MulOneClass M
    ι : Sort u_4
    S : ι → Submonoid M
    C : (x : M) → Membership.mem (iSup fun i => S i) x → Prop
    mem : ∀ (i : ι) (x : M) (hxS : Membership.mem (S i) x), C x ⋯
    one : C 1 ⋯
    mul : ∀ (x y : M) (hx : Membership.mem (iSup fun i => S i) x) (hy : Membership …
    x : M
    hx : Membership.mem (iSup fun i => S i) x
    ⊢ Exists fun Hx => C x Hx
  -/
  refine @iSup_induction _ _ ι S (fun m => ∃ hm, C m hm) _ hx (fun i x hx => ?_) ?_ fun x y => ?_
    /-
      case refine_1
      M : Type u_1
      inst✝ : MulOneClass M
      ι : Sort u_4
      S : ι → Submonoid M
      C : (x : M) → Membership.mem (iSup fun i => S i) x → Prop
      mem : ∀ (i : ι) (x : M) (hxS : Membership.mem (S i) x), C x ⋯
      one : C 1 ⋯
      mul : ∀ (x y : M) (hx : Membership.mem (iSup fun i => S i) x) (hy : Membership …
      x✝ : M
      hx✝ : Membership.mem (iSup fun i => S i) x✝
      i : ι
      x : M
      hx : Membership.mem (S i) x
      ⊢ (fun m => Exists fun hm => C m hm) x
    -/
  · exact ⟨_, mem _ _ hx⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      M : Type u_1
      inst✝ : MulOneClass M
      ι : Sort u_4
      S : ι → Submonoid M
      C : (x : M) → Membership.mem (iSup fun i => S i) x → Prop
      mem : ∀ (i : ι) (x : M) (hxS : Membership.mem (S i) x), C x ⋯
      one : C 1 ⋯
      mul : ∀ (x y : M) (hx : Membership.mem (iSup fun i => S i) x) (hy : Membership …
      x : M
      hx : Membership.mem (iSup fun i => S i) x
      ⊢ (fun m => Exists fun hm => C m hm) 1
    -/
  · exact ⟨_, one⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      M : Type u_1
      inst✝ : MulOneClass M
      ι : Sort u_4
      S : ι → Submonoid M
      C : (x : M) → Membership.mem (iSup fun i => S i) x → Prop
      mem : ∀ (i : ι) (x : M) (hxS : Membership.mem (S i) x), C x ⋯
      one : C 1 ⋯
      mul : ∀ (x y : M) (hx : Membership.mem (iSup fun i => S i) x) (hy : Membership …
      x✝ : M
      hx : Membership.mem (iSup fun i => S i) x✝
      x y : M
      ⊢ (fun m => Exists fun hm => C m hm) x → (fun m => Exists fun hm => C m hm) y  …
    -/
  · rintro ⟨_, Cx⟩ ⟨_, Cy⟩
    /-
      case refine_3.intro.intro
      M : Type u_1
      inst✝ : MulOneClass M
      ι : Sort u_4
      S : ι → Submonoid M
      C : (x : M) → Membership.mem (iSup fun i => S i) x → Prop
      mem : ∀ (i : ι) (x : M) (hxS : Membership.mem (S i) x), C x ⋯
      one : C 1 ⋯
      mul : ∀ (x y : M) (hx : Membership.mem (iSup fun i => S i) x) (hy : Membership …
      x✝ : M
      hx : Membership.mem (iSup fun i => S i) x✝
      x y : M
      w✝¹ : Membership.mem (iSup fun i => S i) x
      Cx : C x w✝¹
      w✝ : Membership.mem (iSup fun i => S i) y
      Cy : C y w✝
      ⊢ Exists fun hm => C (HMul.hMul x y) hm
    -/
    exact ⟨_, mul _ _ _ _ Cx Cy⟩
    /-
      🎉 no goals
    -/


@[to_additive]
theorem closure_range_of : closure (Set.range <| @of α) = ⊤ :=
  eq_top_iff.2 fun x _ =>
    FreeMonoid.recOn x (one_mem _) fun _x _xs hxs =>
      mul_mem (subset_closure <| Set.mem_range_self _) hxs


theorem closure_singleton_eq (x : M) : closure ({x} : Set M) = mrange (powersHom M x) :=
  closure_eq_of_le (Set.singleton_subset_iff.2 ⟨Multiplicative.ofAdd 1, pow_one x⟩) fun _ ⟨_, hn⟩ =>
    hn ▸ pow_mem (subset_closure <| Set.mem_singleton _) _


/-- The submonoid generated by an element of a monoid equals the set of natural number powers of
    the element. -/
theorem mem_closure_singleton {x y : M} : y ∈ closure ({x} : Set M) ↔ ∃ n : ℕ, x ^ n = y := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    x y : M
    ⊢ Iff (Membership.mem (Submonoid.closure (Singleton.singleton x)) y) (Exists f …
  -/
  rw [closure_singleton_eq, mem_mrange]; rfl
                                         /-
                                           🎉 no goals
                                         -/


theorem mem_closure_singleton_self {y : M} : y ∈ closure ({y} : Set M) :=
  mem_closure_singleton.2 ⟨1, pow_one y⟩


theorem closure_singleton_one : closure ({1} : Set M) = ⊥ := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    ⊢ Eq (Submonoid.closure (Singleton.singleton 1)) Bot.bot
  -/
  simp [eq_bot_iff_forall, mem_closure_singleton]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem card_bot {_ : Fintype (⊥ : Submonoid M)} : card (⊥ : Submonoid M) = 1 :=
  card_eq_one_iff.2
    ⟨⟨(1 : M), Set.mem_singleton 1⟩, fun ⟨_y, hy⟩ => Subtype.eq <| mem_bot.1 hy⟩


@[to_additive]
theorem eq_bot_of_card_le (h : card S ≤ 1) : S = ⊥ :=
  let _ := card_le_one_iff_subsingleton.mp h
  eq_bot_of_subsingleton S


@[to_additive]
theorem eq_bot_of_card_eq (h : card S = 1) : S = ⊥ :=
  S.eq_bot_of_card_le (le_of_eq h)


@[to_additive card_le_one_iff_eq_bot]
theorem card_le_one_iff_eq_bot : card S ≤ 1 ↔ S = ⊥ :=
  ⟨fun h =>
    (eq_bot_iff_forall _).2 fun x hx => by
      /-
        M : Type u_1
        inst✝¹ : Monoid M
        S : Submonoid M
        inst✝ : Fintype (Subtype fun x => Membership.mem S x)
        h : LE.le (Fintype.card (Subtype fun x => Membership.mem S x)) 1
        x : M
        hx : Membership.mem S x
        ⊢ Eq x 1
      -/
      simpa [Subtype.ext_iff] using card_le_one_iff.1 h ⟨x, hx⟩ 1,
      /-
        🎉 no goals
      -/
                /-
                  M : Type u_1
                  inst✝¹ : Monoid M
                  S : Submonoid M
                  inst✝ : Fintype (Subtype fun x => Membership.mem S x)
                  h : Eq S Bot.bot
                  ⊢ LE.le (Fintype.card (Subtype fun x => Membership.mem S x)) 1
                -/
    fun h => by simp [h]⟩
                /-
                  🎉 no goals
                -/


@[to_additive]
lemma eq_bot_iff_card : S = ⊥ ↔ card S = 1 :=
      /-
        M : Type u_1
        inst✝¹ : Monoid M
        S : Submonoid M
        inst✝ : Fintype (Subtype fun x => Membership.mem S x)
        ⊢ Eq S Bot.bot → Eq (Fintype.card (Subtype fun x => Membership.mem S x)) 1
      -/
  ⟨by rintro rfl; exact card_bot, eq_bot_of_card_eq⟩
                  /-
                    🎉 no goals
                  -/


@[to_additive]
theorem _root_.FreeMonoid.mrange_lift {α} (f : α → M) :
    mrange (FreeMonoid.lift f) = closure (Set.range f) := by
  rw [mrange_eq_map, ← FreeMonoid.closure_range_of, map_mclosure, ← Set.range_comp,
    FreeMonoid.lift_comp_of]


@[to_additive]
theorem closure_eq_mrange (s : Set M) : closure s = mrange (FreeMonoid.lift ((↑) : s → M)) := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    s : Set M
    ⊢ Eq (Submonoid.closure s) (MonoidHom.mrange (FreeMonoid.lift Subtype.val))
  -/
  rw [FreeMonoid.mrange_lift, Subtype.range_coe]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem closure_eq_image_prod (s : Set M) :
    (closure s : Set M) = List.prod '' { l : List M | ∀ x ∈ l, x ∈ s } := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    s : Set M
    ⊢ Eq (↑(Submonoid.closure s)) (Set.image List.prod (setOf fun l => ∀ (x : M),  …
  -/
  rw [closure_eq_mrange, coe_mrange, ← Set.range_list_map_coe, ← Set.range_comp]
  /-
    M : Type u_1
    inst✝ : Monoid M
    s : Set M
    ⊢ Eq (Set.range ⇑(FreeMonoid.lift Subtype.val)) (Set.range (Function.comp List …
  -/
  exact congrArg _ (funext <| FreeMonoid.lift_apply _)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem exists_list_of_mem_closure {s : Set M} {x : M} (hx : x ∈ closure s) :
    ∃ l : List M, (∀ y ∈ l, y ∈ s) ∧ l.prod = x := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    s : Set M
    x : M
    hx : Membership.mem (Submonoid.closure s) x
    ⊢ Exists fun l => And (∀ (y : M), Membership.mem l y → Membership.mem s y) (Eq …
  -/
  rwa [← SetLike.mem_coe, closure_eq_image_prod, Set.mem_image] at hx
  /-
    🎉 no goals
  -/


@[to_additive]
theorem exists_multiset_of_mem_closure {M : Type*} [CommMonoid M] {s : Set M} {x : M}
    (hx : x ∈ closure s) : ∃ l : Multiset M, (∀ y ∈ l, y ∈ s) ∧ l.prod = x := by
  /-
    M : Type u_4
    inst✝ : CommMonoid M
    s : Set M
    x : M
    hx : Membership.mem (Submonoid.closure s) x
    ⊢ Exists fun l => And (∀ (y : M), Membership.mem l y → Membership.mem s y) (Eq …
  -/
  obtain ⟨l, h1, h2⟩ := exists_list_of_mem_closure hx
  /-
    case intro.intro
    M : Type u_4
    inst✝ : CommMonoid M
    s : Set M
    x : M
    hx : Membership.mem (Submonoid.closure s) x
    l : List M
    h1 : ∀ (y : M), Membership.mem l y → Membership.mem s y
    h2 : Eq l.prod x
    ⊢ Exists fun l => And (∀ (y : M), Membership.mem l y → Membership.mem s y) (Eq …
  -/
  exact ⟨l, h1, (Multiset.prod_coe l).trans h2⟩
  /-
    🎉 no goals
  -/


@[to_additive (attr := elab_as_elim)]
theorem closure_induction_left {s : Set M} {p : (m : M) → m ∈ closure s → Prop}
    (one : p 1 (one_mem _))
    (mul_left : ∀ x (hx : x ∈ s), ∀ (y) hy, p y hy → p (x * y) (mul_mem (subset_closure hx) hy))
    {x : M} (h : x ∈ closure s) :
    p x h := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    s : Set M
    p : (m : M) → Membership.mem (Submonoid.closure s) m → Prop
    one : p 1 ⋯
    mul_left : ∀ (x : M) (hx : Membership.mem s x) (y : M) (hy : Membership.mem (S …
    x : M
    h : Membership.mem (Submonoid.closure s) x
    ⊢ p x h
  -/
  simp_rw [closure_eq_mrange] at h
  /-
    M : Type u_1
    inst✝ : Monoid M
    s : Set M
    p : (m : M) → Membership.mem (Submonoid.closure s) m → Prop
    one : p 1 ⋯
    mul_left : ∀ (x : M) (hx : Membership.mem s x) (y : M) (hy : Membership.mem (S …
    x : M
    h✝ : Membership.mem (Submonoid.closure s) x
    h : Membership.mem (MonoidHom.mrange (FreeMonoid.lift Subtype.val)) x
    ⊢ p x h✝
  -/
  obtain ⟨l, rfl⟩ := h
  induction l using FreeMonoid.inductionOn' with
  | one => exact one
  | mul_of x y ih =>
    simp only [map_mul, FreeMonoid.lift_eval_of]
    refine mul_left _ x.prop (FreeMonoid.lift Subtype.val y) _ (ih ?_)
    simp only [closure_eq_mrange, mem_mrange, exists_apply_eq_apply]


@[to_additive (attr := elab_as_elim)]
theorem induction_of_closure_eq_top_left {s : Set M} {p : M → Prop} (hs : closure s = ⊤) (x : M)
    (one : p 1) (mul : ∀ x ∈ s, ∀ (y), p y → p (x * y)) : p x := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    s : Set M
    p : M → Prop
    hs : Eq (Submonoid.closure s) Top.top
    x : M
    one : p 1
    mul : ∀ (x : M), Membership.mem s x → ∀ (y : M), p y → p (HMul.hMul x y)
    ⊢ p x
  -/
  have : x ∈ closure s := by simp [hs]
  induction this using closure_induction_left with
  | one => exact one
  | mul_left x hx y _ ih => exact mul x hx y ih


@[to_additive (attr := elab_as_elim)]
theorem closure_induction_right {s : Set M} {p : (m : M) → m ∈ closure s → Prop}
    (one : p 1 (one_mem _))
    (mul_right : ∀ x hx, ∀ (y) (hy : y ∈ s), p x hx → p (x * y) (mul_mem hx (subset_closure hy)))
    {x : M} (h : x ∈ closure s) : p x h :=
  closure_induction_left (s := MulOpposite.unop ⁻¹' s)
                                     /-
                                       M : Type u_1
                                       inst✝ : Monoid M
                                       s : Set M
                                       p : (m : M) → Membership.mem (Submonoid.closure s) m → Prop
                                       one : p 1 ⋯
                                       mul_right : ∀ (x : M) (hx : Membership.mem (Submonoid.closure s) x) (y : M) (h …
                                       x : M
                                       h : Membership.mem (Submonoid.closure s) x
                                       m : MulOpposite M
                                       hm : Membership.mem (Submonoid.closure (Set.preimage MulOpposite.unop s)) m
                                       ⊢ Membership.mem (Submonoid.closure s) (MulOpposite.unop m)
                                     -/
    (p := fun m hm => p m.unop <| by rwa [← op_closure] at hm)
                                     /-
                                       🎉 no goals
                                     -/
    one
    (fun _x hx _y _ => mul_right _ _ _ hx)
        /-
          M : Type u_1
          inst✝ : Monoid M
          s : Set M
          p : (m : M) → Membership.mem (Submonoid.closure s) m → Prop
          one : p 1 ⋯
          mul_right : ∀ (x : M) (hx : Membership.mem (Submonoid.closure s) x) (y : M) (h …
          x : M
          h : Membership.mem (Submonoid.closure s) x
          ⊢ Membership.mem (Submonoid.closure (Set.preimage MulOpposite.unop s)) { unop' …
        -/
    (by rwa [← op_closure])
        /-
          🎉 no goals
        -/


@[to_additive (attr := elab_as_elim)]
theorem induction_of_closure_eq_top_right {s : Set M} {p : M → Prop} (hs : closure s = ⊤) (x : M)
    (H1 : p 1) (Hmul : ∀ (x), ∀ y ∈ s, p x → p (x * y)) : p x := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    s : Set M
    p : M → Prop
    hs : Eq (Submonoid.closure s) Top.top
    x : M
    H1 : p 1
    Hmul : ∀ (x y : M), Membership.mem s y → p x → p (HMul.hMul x y)
    ⊢ p x
  -/
  have : x ∈ closure s := by simp [hs]
  induction this using closure_induction_right with
  | one => exact H1
  | mul_right x _ y hy ih => exact Hmul x y hy ih


/-- The submonoid generated by an element. -/
def powers (n : M) : Submonoid M :=
  Submonoid.copy (mrange (powersHom M n)) (Set.range (n ^ · : ℕ → M)) <|
                                              /-
                                                M : Type u_1
                                                A : Type u_2
                                                B : Type u_3
                                                inst✝ : Monoid M
                                                a n✝ n : M
                                                i : Nat
                                                ⊢ Iff (Eq ((fun x => HPow.hPow n✝ x) i) n) (Eq (((powersHom M) n✝) i) n)
                                              -/
    Set.ext fun n => exists_congr fun i => by simp; rfl
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem mem_powers (n : M) : n ∈ powers n :=
  ⟨1, pow_one _⟩


theorem coe_powers (x : M) : ↑(powers x) = Set.range fun n : ℕ => x ^ n :=
  rfl


theorem mem_powers_iff (x z : M) : x ∈ powers z ↔ ∃ n : ℕ, z ^ n = x :=
  Iff.rfl


noncomputable instance decidableMemPowers : DecidablePred (· ∈ Submonoid.powers a) :=
  Classical.decPred _

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO the following instance should follow from a more general principle
-- See also https://github.com/leanprover-community/mathlib4/issues/2417

noncomputable instance fintypePowers [Fintype M] : Fintype (powers a) :=
  inferInstanceAs <| Fintype {y // y ∈ powers a}


theorem powers_eq_closure (n : M) : powers n = closure {n} := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    n : M
    ⊢ Eq (Submonoid.powers n) (Submonoid.closure (Singleton.singleton n))
  -/
  ext
  /-
    case h
    M : Type u_1
    inst✝ : Monoid M
    n x✝ : M
    ⊢ Iff (Membership.mem (Submonoid.powers n) x✝) (Membership.mem (Submonoid.clos …
  -/
  exact mem_closure_singleton.symm
  /-
    🎉 no goals
  -/


                                                                       /-
                                                                         M : Type u_1
                                                                         inst✝ : Monoid M
                                                                         n : M
                                                                         P : Submonoid M
                                                                         ⊢ Iff (LE.le (Submonoid.powers n) P) (Membership.mem P n)
                                                                       -/
lemma powers_le {n : M} {P : Submonoid M} : powers n ≤ P ↔ n ∈ P := by simp [powers_eq_closure]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


lemma powers_one : powers (1 : M) = ⊥ := bot_unique <| powers_le.2 <| one_mem _


theorem _root_.IsIdempotentElem.coe_powers {a : M} (ha : IsIdempotentElem a) :
    (Submonoid.powers a : Set M) = {1, a} :=
  let S : Submonoid M :=
  { carrier := {1, a},
    mul_mem' := by
      /-
        M : Type u_1
        inst✝ : Monoid M
        a : M
        ha : IsIdempotentElem a
        ⊢ ∀ {a_1 b : M}, Membership.mem (Insert.insert 1 (Singleton.singleton a)) a_1  …
      -/
      rintro _ _ (rfl|rfl) (rfl|rfl)
        /-
          case inl.inl
          M : Type u_1
          inst✝ : Monoid M
          a : M
          ha : IsIdempotentElem a
          ⊢ Membership.mem (Insert.insert 1 (Singleton.singleton a)) (HMul.hMul 1 1)
        -/
      · rw [one_mul]; exact .inl rfl
                      /-
                        🎉 no goals
                      -/
        /-
          case inl.inr
          M : Type u_1
          inst✝ : Monoid M
          b✝ : M
          ha : IsIdempotentElem b✝
          ⊢ Membership.mem (Insert.insert 1 (Singleton.singleton b✝)) (HMul.hMul 1 b✝)
        -/
      · rw [one_mul]; exact .inr rfl
                      /-
                        🎉 no goals
                      -/
        /-
          case inr.inl
          M : Type u_1
          inst✝ : Monoid M
          a✝ : M
          ha : IsIdempotentElem a✝
          ⊢ Membership.mem (Insert.insert 1 (Singleton.singleton a✝)) (HMul.hMul a✝ 1)
        -/
      · rw [mul_one]; exact .inr rfl
                      /-
                        🎉 no goals
                      -/
        /-
          case inr.inr
          M : Type u_1
          inst✝ : Monoid M
          b✝ : M
          ha : IsIdempotentElem b✝
          ⊢ Membership.mem (Insert.insert 1 (Singleton.singleton b✝)) (HMul.hMul b✝ b✝)
        -/
      · rw [ha]; exact .inr rfl
                 /-
                   🎉 no goals
                 -/
    one_mem' := .inl rfl }
  suffices Submonoid.powers a = S from congr_arg _ this
  le_antisymm (Submonoid.powers_le.mpr <| .inr rfl)
        /-
          M : Type u_1
          inst✝ : Monoid M
          a : M
          ha : IsIdempotentElem a
          S : Submonoid M := { carrier := Insert.insert 1 (Singleton.singleton a), mul_m …
          ⊢ LE.le S (Submonoid.powers a)
        -/
    (by rintro _ (rfl|rfl); exacts [one_mem _, Submonoid.mem_powers _])
                            /-
                              🎉 no goals
                            -/


/-- The submonoid generated by an element is a group if that element has finite order. -/
abbrev groupPowers {x : M} {n : ℕ} (hpos : 0 < n) (hx : x ^ n = 1) : Group (powers x) where
  inv x := x ^ (n - 1)
  inv_mul_cancel y := Subtype.ext <| by
    /-
      M : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝ : Monoid M
      a x : M
      n : Nat
      hpos : LT.lt 0 n
      hx : Eq (HPow.hPow x n) 1
      y : Subtype fun x_1 => Membership.mem (Submonoid.powers x) x_1
      ⊢ Eq ↑(HMul.hMul (Inv.inv y) y) ↑1
    -/
    obtain ⟨_, k, rfl⟩ := y
    /-
      case mk.intro
      M : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝ : Monoid M
      a x : M
      n : Nat
      hpos : LT.lt 0 n
      hx : Eq (HPow.hPow x n) 1
      k : Nat
      ⊢ Eq ↑(HMul.hMul (Inv.inv ⟨(fun x_1 => HPow.hPow x x_1) k, ⋯⟩) ⟨(fun x_1 => HP …
    -/
    simp only [coe_one, coe_mul, SubmonoidClass.coe_pow]
                     /-
                       M : Type u_1
                       A : Type u_2
                       B : Type u_3
                       inst✝ : Monoid M
                       a x : M
                       n : Nat
                       hpos : LT.lt 0 n
                       hx : Eq (HPow.hPow x n) 1
                       z : Subtype fun x_1 => Membership.mem (Submonoid.powers x) x_1
                       ⊢ Eq ((fun z x_1 => HPow.hPow x_1 (z.natMod ↑n)) 0 z) 1
                     -/
    /-
      case mk.intro
      M : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝ : Monoid M
      a x : M
      n : Nat
      hpos : LT.lt 0 n
      hx : Eq (HPow.hPow x n) 1
      k : Nat
      ⊢ Eq (HMul.hMul (HPow.hPow (HPow.hPow x k) (HSub.hSub n 1)) (HPow.hPow x k)) 1
    -/
                     /-
                       🎉 no goals
                     -/
    rw [← pow_succ, Nat.sub_add_cancel hpos, ← pow_mul, mul_comm, pow_mul, hx, one_pow]
    /-
      M : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝ : Monoid M
      a x✝ : M
      n : Nat
      hpos : LT.lt 0 n
      hx : Eq (HPow.hPow x✝ n) 1
      m : Nat
      x : Subtype fun x => Membership.mem (Submonoid.powers x✝) x
      ⊢ Eq ↑((fun z x => HPow.hPow x (z.natMod ↑n)) (Int.negSucc m) x) ↑(Inv.inv ((f …
    -/
    /-
      🎉 no goals
    -/
    /-
      case mk.intro
      M : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝ : Monoid M
      a x : M
      n : Nat
      hpos : LT.lt 0 n
      hx : Eq (HPow.hPow x n) 1
      m k : Nat
      ⊢ Eq ↑((fun z x_1 => HPow.hPow x_1 (z.natMod ↑n)) (Int.negSucc m) ⟨(fun x_1 => …
    -/
  zpow z x := x ^ z.natMod n
    /-
      case mk.intro
      M : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝ : Monoid M
      a x : M
      n : Nat
      hpos : LT.lt 0 n
      hx : Eq (HPow.hPow x n) 1
      m k : Nat
      ⊢ Eq (HPow.hPow x (HMul.hMul k (HMod.hMod (Int.negSucc m) ↑n).toNat)) (HPow.hP …
    -/
  zpow_zero' z := by simp only [Int.natMod, Int.zero_emod, Int.toNat_zero, pow_zero]
    /-
      case mk.intro
      M : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝ : Monoid M
      a x : M
      n : Nat
      hpos : LT.lt 0 n
      hx : Eq (HPow.hPow x n) 1
      m k : Nat
      ⊢ Eq (HPow.hPow x (HMul.hMul k (HMod.hMod (HAdd.hAdd (Neg.neg ↑(HAdd.hAdd m 1) …
    -/
  zpow_neg' m x := Subtype.ext <| by
    /-
      case mk.intro
      M : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝ : Monoid M
      a x : M
      n : Nat
      hpos : LT.lt 0 n
      hx : Eq (HPow.hPow x n) 1
      m k : Nat
      ⊢ Eq (HPow.hPow x (HMul.hMul k (HMod.hMod (HAdd.hAdd (Neg.neg (HMul.hMul (↑(HA …
    -/
    /-
      M : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝ : Monoid M
      a x✝ : M
      n : Nat
      hpos : LT.lt 0 n
      hx : Eq (HPow.hPow x✝ n) 1
      m : Nat
      x : Subtype fun x => Membership.mem (Submonoid.powers x✝) x
      ⊢ Eq ↑((fun z x => HPow.hPow x (z.natMod ↑n)) (↑m.succ) x) ↑(HMul.hMul ((fun z …
    -/
    obtain ⟨_, k, rfl⟩ := x
    /-
      case mk.intro
      M : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝ : Monoid M
      a x : M
      n : Nat
      hpos : LT.lt 0 n
      hx : Eq (HPow.hPow x n) 1
      m k : Nat
      ⊢ Eq ↑((fun z x_1 => HPow.hPow x_1 (z.natMod ↑n)) ↑m.succ ⟨(fun x_1 => HPow.hP …
    -/
    /-
      case mk.intro
      M : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝ : Monoid M
      a x : M
      n : Nat
      hpos : LT.lt 0 n
      hx : Eq (HPow.hPow x n) 1
      m k : Nat
      ⊢ Eq (HPow.hPow x (HMul.hMul k (↑(HMod.hMod (HMul.hMul (HAdd.hAdd m 1) (HSub.h …
    -/
    /-
      case mk.intro
      M : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝ : Monoid M
      a x : M
      n : Nat
      hpos : LT.lt 0 n
      hx : Eq (HPow.hPow x n) 1
      m k : Nat
      ⊢ Eq (HPow.hPow x (HMul.hMul k (HMod.hMod ↑m.succ ↑n).toNat)) (HMul.hMul (HPow …
    -/
    simp only [← pow_mul, Int.natMod, SubmonoidClass.coe_pow]
    /-
      case mk.intro
      M : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝ : Monoid M
      a x : M
      n : Nat
      hpos : LT.lt 0 n
      hx : Eq (HPow.hPow x n) 1
      m k : Nat
      ⊢ Eq (HPow.hPow x (HMul.hMul k (↑(HMod.hMod m.succ n)).toNat)) (HMul.hMul (HPo …
    -/
    rw [Int.negSucc_coe, ← Int.add_mul_emod_self (b := (m + 1 : ℕ))]
    /-
      case mk.intro
      M : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝ : Monoid M
      a x : M
      n : Nat
      hpos : LT.lt 0 n
      hx : Eq (HPow.hPow x n) 1
      m k : Nat
      ⊢ Eq (HPow.hPow (HPow.hPow x m.succ) k) (HMul.hMul (HPow.hPow (HPow.hPow x m)  …
    -/
    nth_rw 1 [← mul_one ((m + 1 : ℕ) : ℤ)]
    /-
      🎉 no goals
    -/
    rw [← sub_eq_neg_add, ← mul_sub, ← Int.natCast_pred_of_pos hpos]; norm_cast
    simp only [Int.toNat_natCast]
    rw [mul_comm, pow_mul, ← pow_eq_pow_mod _ hx, mul_comm k, mul_assoc, pow_mul _ (_ % _),
      ← pow_eq_pow_mod _ hx, pow_mul, pow_mul]
  zpow_succ' m x := Subtype.ext <| by
    obtain ⟨_, k, rfl⟩ := x
    simp only [← pow_mul, Int.natMod, SubmonoidClass.coe_pow, coe_mul]
    norm_cast
    iterate 2 rw [Int.toNat_natCast, mul_comm, pow_mul, ← pow_eq_pow_mod _ hx]
    rw [← pow_mul _ m, mul_comm, pow_mul, ← pow_succ, ← pow_mul, mul_comm, pow_mul]


/-- Exponentiation map from natural numbers to powers. -/
@[simps!]
def pow (n : M) (m : ℕ) : powers n :=
  (powersHom M n).mrangeRestrict (Multiplicative.ofAdd m)


theorem pow_apply (n : M) (m : ℕ) : Submonoid.pow n m = ⟨n ^ m, m, rfl⟩ :=
  rfl


/-- Logarithms from powers to natural numbers. -/
def log [DecidableEq M] {n : M} (p : powers n) : ℕ :=
  Nat.find <| (mem_powers_iff p.val n).mp p.prop


@[simp]
theorem pow_log_eq_self [DecidableEq M] {n : M} (p : powers n) : pow n (log p) = p :=
  Subtype.ext <| Nat.find_spec p.prop


theorem pow_right_injective_iff_pow_injective {n : M} :
    (Function.Injective fun m : ℕ => n ^ m) ↔ Function.Injective (pow n) :=
  Subtype.coe_injective.of_comp_iff (pow n)


@[simp]
theorem log_pow_eq_self [DecidableEq M] {n : M} (h : Function.Injective fun m : ℕ => n ^ m)
    (m : ℕ) : log (pow n m) = m :=
  pow_right_injective_iff_pow_injective.mp h <| pow_log_eq_self _


/-- The exponentiation map is an isomorphism from the additive monoid on natural numbers to powers
when it is injective. The inverse is given by the logarithms. -/
@[simps]
def powLogEquiv [DecidableEq M] {n : M} (h : Function.Injective fun m : ℕ => n ^ m) :
    Multiplicative ℕ ≃* powers n where
  toFun m := pow n m.toAdd
  invFun m := Multiplicative.ofAdd (log m)
  left_inv := log_pow_eq_self h
  right_inv := pow_log_eq_self
                     /-
                       M : Type u_1
                       A : Type u_2
                       B : Type u_3
                       inst✝¹ : Monoid M
                       a : M
                       inst✝ : DecidableEq M
                       n : M
                       h : Function.Injective fun m => HPow.hPow n m
                       x✝¹ x✝ : Multiplicative Nat
                       ⊢ Eq ({ toFun := fun m => Submonoid.pow n (Multiplicative.toAdd m), invFun :=  …
                     -/
  map_mul' _ _ := by simp only [pow, map_mul, ofAdd_add, toAdd_mul]
                     /-
                       🎉 no goals
                     -/


theorem log_mul [DecidableEq M] {n : M} (h : Function.Injective fun m : ℕ => n ^ m)
    (x y : powers (n : M)) : log (x * y) = log x + log y :=
  map_mul (powLogEquiv h).symm x y


theorem log_pow_int_eq_self {x : ℤ} (h : 1 < x.natAbs) (m : ℕ) : log (pow x m) = m :=
  (powLogEquiv (Int.pow_right_injective h)).symm_apply_apply _


@[simp]
theorem map_powers {N : Type*} {F : Type*} [Monoid N] [FunLike F M N] [MonoidHomClass F M N]
    (f : F) (m : M) :
    (powers m).map f = powers (f m) := by
  /-
    M : Type u_1
    inst✝³ : Monoid M
    N : Type u_4
    F : Type u_5
    inst✝² : Monoid N
    inst✝¹ : FunLike F M N
    inst✝ : MonoidHomClass F M N
    f : F
    m : M
    ⊢ Eq (Submonoid.map f (Submonoid.powers m)) (Submonoid.powers (f m))
  -/
  simp only [powers_eq_closure, map_mclosure f, Set.image_singleton]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsScalarTower.of_mclosure_eq_top {N α} [Monoid M] [MulAction M N] [SMul N α] [MulAction M α]
    {s : Set M} (htop : Submonoid.closure s = ⊤)
    (hs : ∀ x ∈ s, ∀ (y : N) (z : α), (x • y) • z = x • y • z) : IsScalarTower M N α := by
  /-
    M : Type u_1
    N : Type u_4
    α : Type u_5
    inst✝³ : Monoid M
    inst✝² : MulAction M N
    inst✝¹ : SMul N α
    inst✝ : MulAction M α
    s : Set M
    htop : Eq (Submonoid.closure s) Top.top
    hs : ∀ (x : M), Membership.mem s x → ∀ (y : N) (z : α), Eq (HSMul.hSMul (HSMul …
    ⊢ IsScalarTower M N α
  -/
  refine ⟨fun x => Submonoid.induction_of_closure_eq_top_left htop x ?_ ?_⟩
    /-
      case refine_1
      M : Type u_1
      N : Type u_4
      α : Type u_5
      inst✝³ : Monoid M
      inst✝² : MulAction M N
      inst✝¹ : SMul N α
      inst✝ : MulAction M α
      s : Set M
      htop : Eq (Submonoid.closure s) Top.top
      hs : ∀ (x : M), Membership.mem s x → ∀ (y : N) (z : α), Eq (HSMul.hSMul (HSMul …
      x : M
      ⊢ ∀ (y : N) (z : α), Eq (HSMul.hSMul (HSMul.hSMul 1 y) z) (HSMul.hSMul 1 (HSMu …
    -/
  · intro y z
    /-
      case refine_1
      M : Type u_1
      N : Type u_4
      α : Type u_5
      inst✝³ : Monoid M
      inst✝² : MulAction M N
      inst✝¹ : SMul N α
      inst✝ : MulAction M α
      s : Set M
      htop : Eq (Submonoid.closure s) Top.top
      hs : ∀ (x : M), Membership.mem s x → ∀ (y : N) (z : α), Eq (HSMul.hSMul (HSMul …
      x : M
      y : N
      z : α
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul 1 y) z) (HSMul.hSMul 1 (HSMul.hSMul y z))
    -/
    rw [one_smul, one_smul]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      M : Type u_1
      N : Type u_4
      α : Type u_5
      inst✝³ : Monoid M
      inst✝² : MulAction M N
      inst✝¹ : SMul N α
      inst✝ : MulAction M α
      s : Set M
      htop : Eq (Submonoid.closure s) Top.top
      hs : ∀ (x : M), Membership.mem s x → ∀ (y : N) (z : α), Eq (HSMul.hSMul (HSMul …
      x : M
      ⊢ ∀ (x : M), Membership.mem s x → ∀ (y : M), (∀ (y_1 : N) (z : α), Eq (HSMul.h …
    -/
  · clear x
    /-
      case refine_2
      M : Type u_1
      N : Type u_4
      α : Type u_5
      inst✝³ : Monoid M
      inst✝² : MulAction M N
      inst✝¹ : SMul N α
      inst✝ : MulAction M α
      s : Set M
      htop : Eq (Submonoid.closure s) Top.top
      hs : ∀ (x : M), Membership.mem s x → ∀ (y : N) (z : α), Eq (HSMul.hSMul (HSMul …
      ⊢ ∀ (x : M), Membership.mem s x → ∀ (y : M), (∀ (y_1 : N) (z : α), Eq (HSMul.h …
    -/
    intro x hx x' hx' y z
    /-
      case refine_2
      M : Type u_1
      N : Type u_4
      α : Type u_5
      inst✝³ : Monoid M
      inst✝² : MulAction M N
      inst✝¹ : SMul N α
      inst✝ : MulAction M α
      s : Set M
      htop : Eq (Submonoid.closure s) Top.top
      hs : ∀ (x : M), Membership.mem s x → ∀ (y : N) (z : α), Eq (HSMul.hSMul (HSMul …
      x : M
      hx : Membership.mem s x
      x' : M
      hx' : ∀ (y : N) (z : α), Eq (HSMul.hSMul (HSMul.hSMul x' y) z) (HSMul.hSMul x' …
      y : N
      z : α
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul (HMul.hMul x x') y) z) (HSMul.hSMul (HMul.hMul  …
    -/
    rw [mul_smul, mul_smul, hs x hx, hx']
    /-
      🎉 no goals
    -/


@[to_additive]
theorem SMulCommClass.of_mclosure_eq_top {N α} [Monoid M] [SMul N α] [MulAction M α] {s : Set M}
    (htop : Submonoid.closure s = ⊤) (hs : ∀ x ∈ s, ∀ (y : N) (z : α), x • y • z = y • x • z) :
    SMulCommClass M N α := by
  /-
    M : Type u_1
    N : Type u_4
    α : Type u_5
    inst✝² : Monoid M
    inst✝¹ : SMul N α
    inst✝ : MulAction M α
    s : Set M
    htop : Eq (Submonoid.closure s) Top.top
    hs : ∀ (x : M), Membership.mem s x → ∀ (y : N) (z : α), Eq (HSMul.hSMul x (HSM …
    ⊢ SMulCommClass M N α
  -/
  refine ⟨fun x => Submonoid.induction_of_closure_eq_top_left htop x ?_ ?_⟩
    /-
      case refine_1
      M : Type u_1
      N : Type u_4
      α : Type u_5
      inst✝² : Monoid M
      inst✝¹ : SMul N α
      inst✝ : MulAction M α
      s : Set M
      htop : Eq (Submonoid.closure s) Top.top
      hs : ∀ (x : M), Membership.mem s x → ∀ (y : N) (z : α), Eq (HSMul.hSMul x (HSM …
      x : M
      ⊢ ∀ (n : N) (a : α), Eq (HSMul.hSMul 1 (HSMul.hSMul n a)) (HSMul.hSMul n (HSMu …
    -/
  · intro y z
    /-
      case refine_1
      M : Type u_1
      N : Type u_4
      α : Type u_5
      inst✝² : Monoid M
      inst✝¹ : SMul N α
      inst✝ : MulAction M α
      s : Set M
      htop : Eq (Submonoid.closure s) Top.top
      hs : ∀ (x : M), Membership.mem s x → ∀ (y : N) (z : α), Eq (HSMul.hSMul x (HSM …
      x : M
      y : N
      z : α
      ⊢ Eq (HSMul.hSMul 1 (HSMul.hSMul y z)) (HSMul.hSMul y (HSMul.hSMul 1 z))
    -/
    rw [one_smul, one_smul]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      M : Type u_1
      N : Type u_4
      α : Type u_5
      inst✝² : Monoid M
      inst✝¹ : SMul N α
      inst✝ : MulAction M α
      s : Set M
      htop : Eq (Submonoid.closure s) Top.top
      hs : ∀ (x : M), Membership.mem s x → ∀ (y : N) (z : α), Eq (HSMul.hSMul x (HSM …
      x : M
      ⊢ ∀ (x : M), Membership.mem s x → ∀ (y : M), (∀ (n : N) (a : α), Eq (HSMul.hSM …
    -/
  · clear x
    /-
      case refine_2
      M : Type u_1
      N : Type u_4
      α : Type u_5
      inst✝² : Monoid M
      inst✝¹ : SMul N α
      inst✝ : MulAction M α
      s : Set M
      htop : Eq (Submonoid.closure s) Top.top
      hs : ∀ (x : M), Membership.mem s x → ∀ (y : N) (z : α), Eq (HSMul.hSMul x (HSM …
      ⊢ ∀ (x : M), Membership.mem s x → ∀ (y : M), (∀ (n : N) (a : α), Eq (HSMul.hSM …
    -/
    intro x hx x' hx' y z
    /-
      case refine_2
      M : Type u_1
      N : Type u_4
      α : Type u_5
      inst✝² : Monoid M
      inst✝¹ : SMul N α
      inst✝ : MulAction M α
      s : Set M
      htop : Eq (Submonoid.closure s) Top.top
      hs : ∀ (x : M), Membership.mem s x → ∀ (y : N) (z : α), Eq (HSMul.hSMul x (HSM …
      x : M
      hx : Membership.mem s x
      x' : M
      hx' : ∀ (n : N) (a : α), Eq (HSMul.hSMul x' (HSMul.hSMul n a)) (HSMul.hSMul n  …
      y : N
      z : α
      ⊢ Eq (HSMul.hSMul (HMul.hMul x x') (HSMul.hSMul y z)) (HSMul.hSMul y (HSMul.hS …
    -/
    rw [mul_smul, mul_smul, hx', hs x hx]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem sup_eq_range (s t : Submonoid N) : s ⊔ t = mrange (s.subtype.coprod t.subtype) := by
  rw [mrange_eq_map, ← mrange_inl_sup_mrange_inr, map_sup, map_mrange, coprod_comp_inl, map_mrange,
    coprod_comp_inr, mrange_subtype, mrange_subtype]


@[to_additive]
theorem mem_sup {s t : Submonoid N} {x : N} : x ∈ s ⊔ t ↔ ∃ y ∈ s, ∃ z ∈ t, y * z = x := by
  simp only [sup_eq_range, mem_mrange, coprod_apply, coe_subtype, Prod.exists,
    Subtype.exists, exists_prop]


theorem closure_singleton_eq (x : A) :
    closure ({x} : Set A) = AddMonoidHom.mrange (multiplesHom A x) :=
  closure_eq_of_le (Set.singleton_subset_iff.2 ⟨1, one_nsmul x⟩) fun _ ⟨_n, hn⟩ =>
    hn ▸ nsmul_mem (subset_closure <| Set.mem_singleton _) _


/-- The `AddSubmonoid` generated by an element of an `AddMonoid` equals the set of
natural number multiples of the element. -/
theorem mem_closure_singleton {x y : A} : y ∈ closure ({x} : Set A) ↔ ∃ n : ℕ, n • x = y := by
  /-
    A : Type u_2
    inst✝ : AddMonoid A
    x y : A
    ⊢ Iff (Membership.mem (AddSubmonoid.closure (Singleton.singleton x)) y) (Exist …
  -/
  rw [closure_singleton_eq, AddMonoidHom.mem_mrange]; rfl
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem closure_singleton_zero : closure ({0} : Set A) = ⊥ := by
  /-
    A : Type u_2
    inst✝ : AddMonoid A
    ⊢ Eq (AddSubmonoid.closure (Singleton.singleton 0)) Bot.bot
  -/
  simp [eq_bot_iff_forall, mem_closure_singleton, nsmul_zero]
  /-
    🎉 no goals
  -/


/-- The additive submonoid generated by an element. -/
def multiples (x : A) : AddSubmonoid A :=
  AddSubmonoid.copy (AddMonoidHom.mrange (multiplesHom A x)) (Set.range (fun i => i • x : ℕ → A)) <|
                                              /-
                                                M : Type u_1
                                                A : Type u_2
                                                B : Type u_3
                                                inst✝ : AddMonoid A
                                                x n : A
                                                i : Nat
                                                ⊢ Iff (Eq ((fun i => HSMul.hSMul i x) i) n) (Eq (((multiplesHom A) x) i) n)
                                              -/
    Set.ext fun n => exists_congr fun i => by simp
                                              /-
                                                🎉 no goals
                                              -/


attribute [to_additive existing] Submonoid.powers


attribute [to_additive (attr := simp)] Submonoid.mem_powers


attribute [to_additive (attr := norm_cast)] Submonoid.coe_powers


attribute [to_additive] Submonoid.mem_powers_iff


attribute [to_additive] Submonoid.decidableMemPowers


attribute [to_additive] Submonoid.fintypePowers


attribute [to_additive] Submonoid.powers_eq_closure


attribute [to_additive] Submonoid.powers_le


attribute [to_additive (attr := simp)] Submonoid.powers_one


attribute [to_additive "The additive submonoid generated by an element is
an additive group if that element has finite order."] Submonoid.groupPowers


/-- The product of an element of the additive closure of a multiplicative subsemigroup `M`
and an element of `M` is contained in the additive closure of `M`. -/
theorem mul_right_mem_add_closure (ha : a ∈ AddSubmonoid.closure (S : Set R)) (hb : b ∈ S) :
    a * b ∈ AddSubmonoid.closure (S : Set R) := by
  induction ha using AddSubmonoid.closure_induction with
  | mem r hr => exact AddSubmonoid.mem_closure.mpr fun y hy => hy (mul_mem hr hb)
  | one => simp only [zero_mul, zero_mem _]
  | mul r s _ _ hr hs => simpa only [add_mul] using add_mem hr hs


/-- The product of two elements of the additive closure of a submonoid `M` is an element of the
additive closure of `M`. -/
theorem mul_mem_add_closure (ha : a ∈ AddSubmonoid.closure (S : Set R))
    (hb : b ∈ AddSubmonoid.closure (S : Set R)) : a * b ∈ AddSubmonoid.closure (S : Set R) := by
  induction hb using AddSubmonoid.closure_induction with
  | mem r hr => exact MulMemClass.mul_right_mem_add_closure ha hr
  | one => simp only [mul_zero, zero_mem _]
  | mul r s _ _ hr hs => simpa only [mul_add] using add_mem hr hs


/-- The product of an element of `S` and an element of the additive closure of a multiplicative
submonoid `S` is contained in the additive closure of `S`. -/
theorem mul_left_mem_add_closure (ha : a ∈ S) (hb : b ∈ AddSubmonoid.closure (S : Set R)) :
    a * b ∈ AddSubmonoid.closure (S : Set R) :=
  mul_mem_add_closure (AddSubmonoid.mem_closure.mpr fun _sT hT => hT ha) hb


/-- An element is in the closure of a two-element set if it is a linear combination of those two
elements. -/
@[to_additive
      "An element is in the closure of a two-element set if it is a linear combination of
      those two elements."]
theorem mem_closure_pair {A : Type*} [CommMonoid A] (a b c : A) :
    c ∈ Submonoid.closure ({a, b} : Set A) ↔ ∃ m n : ℕ, a ^ m * b ^ n = c := by
  /-
    A : Type u_4
    inst✝ : CommMonoid A
    a b c : A
    ⊢ Iff (Membership.mem (Submonoid.closure (Insert.insert a (Singleton.singleton …
  -/
  rw [← Set.singleton_union, Submonoid.closure_union, mem_sup]
  /-
    A : Type u_4
    inst✝ : CommMonoid A
    a b c : A
    ⊢ Iff (Exists fun y => And (Membership.mem (Submonoid.closure (Singleton.singl …
  -/
  simp_rw [mem_closure_singleton, exists_exists_eq_and]
  /-
    🎉 no goals
  -/


theorem ofMul_image_powers_eq_multiples_ofMul [Monoid M] {x : M} :
    Additive.ofMul '' (Submonoid.powers x : Set M) = AddSubmonoid.multiples (Additive.ofMul x) := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    x : M
    ⊢ Eq (Set.image ⇑Additive.ofMul ↑(Submonoid.powers x)) ↑(AddSubmonoid.multiple …
  -/
  ext
  /-
    case h
    M : Type u_1
    inst✝ : Monoid M
    x : M
    x✝ : Additive M
    ⊢ Iff (Membership.mem (Set.image ⇑Additive.ofMul ↑(Submonoid.powers x)) x✝) (M …
  -/
  constructor
    /-
      case h.mp
      M : Type u_1
      inst✝ : Monoid M
      x : M
      x✝ : Additive M
      ⊢ Membership.mem (Set.image ⇑Additive.ofMul ↑(Submonoid.powers x)) x✝ → Member …
    -/
  · rintro ⟨y, ⟨n, hy1⟩, hy2⟩
    /-
      case h.mp.intro.intro.intro
      M : Type u_1
      inst✝ : Monoid M
      x : M
      x✝ : Additive M
      y : M
      hy2 : Eq (Additive.ofMul y) x✝
      n : Nat
      hy1 : Eq ((fun x_1 => HPow.hPow x x_1) n) y
      ⊢ Membership.mem (↑(AddSubmonoid.multiples (Additive.ofMul x))) x✝
    -/
    use n
    /-
      case h
      M : Type u_1
      inst✝ : Monoid M
      x : M
      x✝ : Additive M
      y : M
      hy2 : Eq (Additive.ofMul y) x✝
      n : Nat
      hy1 : Eq ((fun x_1 => HPow.hPow x x_1) n) y
      ⊢ Eq ((fun i => HSMul.hSMul i (Additive.ofMul x)) n) x✝
    -/
    simpa [← ofMul_pow, hy1]
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      M : Type u_1
      inst✝ : Monoid M
      x : M
      x✝ : Additive M
      ⊢ Membership.mem (↑(AddSubmonoid.multiples (Additive.ofMul x))) x✝ → Membershi …
    -/
  · rintro ⟨n, hn⟩
    /-
      case h.mpr.intro
      M : Type u_1
      inst✝ : Monoid M
      x : M
      x✝ : Additive M
      n : Nat
      hn : Eq ((fun i => HSMul.hSMul i (Additive.ofMul x)) n) x✝
      ⊢ Membership.mem (Set.image ⇑Additive.ofMul ↑(Submonoid.powers x)) x✝
    -/
    refine ⟨x ^ n, ⟨n, rfl⟩, ?_⟩
    /-
      case h.mpr.intro
      M : Type u_1
      inst✝ : Monoid M
      x : M
      x✝ : Additive M
      n : Nat
      hn : Eq ((fun i => HSMul.hSMul i (Additive.ofMul x)) n) x✝
      ⊢ Eq (Additive.ofMul (HPow.hPow x n)) x✝
    -/
    rwa [ofMul_pow]
    /-
      🎉 no goals
    -/


theorem ofAdd_image_multiples_eq_powers_ofAdd [AddMonoid A] {x : A} :
    Multiplicative.ofAdd '' (AddSubmonoid.multiples x : Set A) =
      Submonoid.powers (Multiplicative.ofAdd x) := by
  /-
    A : Type u_2
    inst✝ : AddMonoid A
    x : A
    ⊢ Eq (Set.image ⇑Multiplicative.ofAdd ↑(AddSubmonoid.multiples x)) ↑(Submonoid …
  -/
  symm
  /-
    A : Type u_2
    inst✝ : AddMonoid A
    x : A
    ⊢ Eq (↑(Submonoid.powers (Multiplicative.ofAdd x))) (Set.image ⇑Multiplicative …
  -/
  rw [Equiv.eq_image_iff_symm_image_eq]
  /-
    A : Type u_2
    inst✝ : AddMonoid A
    x : A
    ⊢ Eq (Set.image ⇑Multiplicative.ofAdd.symm ↑(Submonoid.powers (Multiplicative. …
  -/
  exact ofMul_image_powers_eq_multiples_ofMul
  /-
    🎉 no goals
  -/


/-- The submonoid of primal elements in a cancellative commutative monoid with zero. -/
def Submonoid.isPrimal (α) [CancelCommMonoidWithZero α] : Submonoid α where
  carrier := {a | IsPrimal a}
  mul_mem' := IsPrimal.mul
  one_mem' := isUnit_one.isPrimal


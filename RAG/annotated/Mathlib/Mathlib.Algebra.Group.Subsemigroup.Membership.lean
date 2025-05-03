@[to_additive]
theorem mem_iSup_of_directed {S : ι → Subsemigroup M} (hS : Directed (· ≤ ·) S) {x : M} :
    (x ∈ ⨆ i, S i) ↔ ∃ i, x ∈ S i := by
  /-
    ι : Sort u_1
    M : Type u_2
    inst✝ : Mul M
    S : ι → Subsemigroup M
    hS : Directed (fun x1 x2 => LE.le x1 x2) S
    x : M
    ⊢ Iff (Membership.mem (iSup fun i => S i) x) (Exists fun i => Membership.mem ( …
  -/
  refine ⟨?_, fun ⟨i, hi⟩ ↦ le_iSup S i hi⟩
  suffices x ∈ closure (⋃ i, (S i : Set M)) → ∃ i, x ∈ S i by
    simpa only [closure_iUnion, closure_eq (S _)] using this
  /-
    ι : Sort u_1
    M : Type u_2
    inst✝ : Mul M
    S : ι → Subsemigroup M
    hS : Directed (fun x1 x2 => LE.le x1 x2) S
    x : M
    ⊢ Membership.mem (Subsemigroup.closure (Set.iUnion fun i => ↑(S i))) x → Exist …
  -/
  refine fun hx ↦ closure_induction (fun y hy ↦ mem_iUnion.mp hy) ?_ hx
  /-
    ι : Sort u_1
    M : Type u_2
    inst✝ : Mul M
    S : ι → Subsemigroup M
    hS : Directed (fun x1 x2 => LE.le x1 x2) S
    x : M
    hx : Membership.mem (Subsemigroup.closure (Set.iUnion fun i => ↑(S i))) x
    ⊢ ∀ (x y : M), Membership.mem (Subsemigroup.closure (Set.iUnion fun i => ↑(S i …
  -/
  rintro x y - - ⟨i, hi⟩ ⟨j, hj⟩
  /-
    case intro.intro
    ι : Sort u_1
    M : Type u_2
    inst✝ : Mul M
    S : ι → Subsemigroup M
    hS : Directed (fun x1 x2 => LE.le x1 x2) S
    x✝ : M
    hx : Membership.mem (Subsemigroup.closure (Set.iUnion fun i => ↑(S i))) x✝
    x y : M
    i : ι
    hi : Membership.mem (S i) x
    j : ι
    hj : Membership.mem (S j) y
    ⊢ Exists fun i => Membership.mem (S i) (HMul.hMul x y)
  -/
  rcases hS i j with ⟨k, hki, hkj⟩
  /-
    case intro.intro.intro.intro
    ι : Sort u_1
    M : Type u_2
    inst✝ : Mul M
    S : ι → Subsemigroup M
    hS : Directed (fun x1 x2 => LE.le x1 x2) S
    x✝ : M
    hx : Membership.mem (Subsemigroup.closure (Set.iUnion fun i => ↑(S i))) x✝
    x y : M
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
theorem coe_iSup_of_directed {S : ι → Subsemigroup M} (hS : Directed (· ≤ ·) S) :
    ((⨆ i, S i : Subsemigroup M) : Set M) = ⋃ i, S i :=
                      /-
                        ι : Sort u_1
                        M : Type u_2
                        inst✝ : Mul M
                        S : ι → Subsemigroup M
                        hS : Directed (fun x1 x2 => LE.le x1 x2) S
                        x : M
                        ⊢ Iff (Membership.mem (↑(iSup fun i => S i)) x) (Membership.mem (Set.iUnion fu …
                      -/
  Set.ext fun x => by simp [mem_iSup_of_directed hS]
                      /-
                        🎉 no goals
                      -/


@[to_additive]
theorem mem_sSup_of_directed_on {S : Set (Subsemigroup M)} (hS : DirectedOn (· ≤ ·) S) {x : M} :
    x ∈ sSup S ↔ ∃ s ∈ S, x ∈ s := by
  simp only [sSup_eq_iSup', mem_iSup_of_directed hS.directed_val, SetCoe.exists, Subtype.coe_mk,
    exists_prop]


@[to_additive]
theorem coe_sSup_of_directed_on {S : Set (Subsemigroup M)} (hS : DirectedOn (· ≤ ·) S) :
    (↑(sSup S) : Set M) = ⋃ s ∈ S, ↑s :=
                      /-
                        M : Type u_2
                        inst✝ : Mul M
                        S : Set (Subsemigroup M)
                        hS : DirectedOn (fun x1 x2 => LE.le x1 x2) S
                        x : M
                        ⊢ Iff (Membership.mem (↑(SupSet.sSup S)) x) (Membership.mem (Set.iUnion fun s  …
                      -/
  Set.ext fun x => by simp [mem_sSup_of_directed_on hS]
                      /-
                        🎉 no goals
                      -/


@[to_additive]
theorem mem_sup_left {S T : Subsemigroup M} : ∀ {x : M}, x ∈ S → x ∈ S ⊔ T := by
  /-
    M : Type u_2
    inst✝ : Mul M
    S T : Subsemigroup M
    ⊢ ∀ {x : M}, Membership.mem S x → Membership.mem (Max.max S T) x
  -/
  have : S ≤ S ⊔ T := le_sup_left
  /-
    M : Type u_2
    inst✝ : Mul M
    S T : Subsemigroup M
    this : LE.le S (Max.max S T)
    ⊢ ∀ {x : M}, Membership.mem S x → Membership.mem (Max.max S T) x
  -/
  tauto
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mem_sup_right {S T : Subsemigroup M} : ∀ {x : M}, x ∈ T → x ∈ S ⊔ T := by
  /-
    M : Type u_2
    inst✝ : Mul M
    S T : Subsemigroup M
    ⊢ ∀ {x : M}, Membership.mem T x → Membership.mem (Max.max S T) x
  -/
  have : T ≤ S ⊔ T := le_sup_right
  /-
    M : Type u_2
    inst✝ : Mul M
    S T : Subsemigroup M
    this : LE.le T (Max.max S T)
    ⊢ ∀ {x : M}, Membership.mem T x → Membership.mem (Max.max S T) x
  -/
  tauto
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_mem_sup {S T : Subsemigroup M} {x y : M} (hx : x ∈ S) (hy : y ∈ T) : x * y ∈ S ⊔ T :=
  mul_mem (mem_sup_left hx) (mem_sup_right hy)


@[to_additive]
theorem mem_iSup_of_mem {S : ι → Subsemigroup M} (i : ι) : ∀ {x : M}, x ∈ S i → x ∈ iSup S := by
  /-
    ι : Sort u_1
    M : Type u_2
    inst✝ : Mul M
    S : ι → Subsemigroup M
    i : ι
    ⊢ ∀ {x : M}, Membership.mem (S i) x → Membership.mem (iSup S) x
  -/
  have : S i ≤ iSup S := le_iSup _ _
  /-
    ι : Sort u_1
    M : Type u_2
    inst✝ : Mul M
    S : ι → Subsemigroup M
    i : ι
    this : LE.le (S i) (iSup S)
    ⊢ ∀ {x : M}, Membership.mem (S i) x → Membership.mem (iSup S) x
  -/
  tauto
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mem_sSup_of_mem {S : Set (Subsemigroup M)} {s : Subsemigroup M} (hs : s ∈ S) :
    ∀ {x : M}, x ∈ s → x ∈ sSup S := by
  /-
    M : Type u_2
    inst✝ : Mul M
    S : Set (Subsemigroup M)
    s : Subsemigroup M
    hs : Membership.mem S s
    ⊢ ∀ {x : M}, Membership.mem s x → Membership.mem (SupSet.sSup S) x
  -/
  have : s ≤ sSup S := le_sSup hs
  /-
    M : Type u_2
    inst✝ : Mul M
    S : Set (Subsemigroup M)
    s : Subsemigroup M
    hs : Membership.mem S s
    this : LE.le s (SupSet.sSup S)
    ⊢ ∀ {x : M}, Membership.mem s x → Membership.mem (SupSet.sSup S) x
  -/
  tauto
  /-
    🎉 no goals
  -/


/-- An induction principle for elements of `⨆ i, S i`.
If `C` holds all elements of `S i` for all `i`, and is preserved under multiplication,
then it holds for all elements of the supremum of `S`. -/
@[to_additive (attr := elab_as_elim)
"An induction principle for elements of `⨆ i, S i`. If `C` holds all
elements of `S i` for all `i`, and is preserved under addition, then it holds for all elements of
the supremum of `S`."]
theorem iSup_induction (S : ι → Subsemigroup M) {C : M → Prop} {x₁ : M} (hx₁ : x₁ ∈ ⨆ i, S i)
    (mem : ∀ i, ∀ x₂ ∈ S i, C x₂) (mul : ∀ x y, C x → C y → C (x * y)) : C x₁ := by
  /-
    ι : Sort u_1
    M : Type u_2
    inst✝ : Mul M
    S : ι → Subsemigroup M
    C : M → Prop
    x₁ : M
    hx₁ : Membership.mem (iSup fun i => S i) x₁
    mem : ∀ (i : ι) (x₂ : M), Membership.mem (S i) x₂ → C x₂
    mul : ∀ (x y : M), C x → C y → C (HMul.hMul x y)
    ⊢ C x₁
  -/
  rw [iSup_eq_closure] at hx₁
  /-
    ι : Sort u_1
    M : Type u_2
    inst✝ : Mul M
    S : ι → Subsemigroup M
    C : M → Prop
    x₁ : M
    hx₁ : Membership.mem (Subsemigroup.closure (Set.iUnion fun i => ↑(S i))) x₁
    mem : ∀ (i : ι) (x₂ : M), Membership.mem (S i) x₂ → C x₂
    mul : ∀ (x y : M), C x → C y → C (HMul.hMul x y)
    ⊢ C x₁
  -/
  refine closure_induction (fun x₂ hx₂ => ?_) (fun x y _ _ ↦ mul x y) hx₁
  /-
    ι : Sort u_1
    M : Type u_2
    inst✝ : Mul M
    S : ι → Subsemigroup M
    C : M → Prop
    x₁ : M
    hx₁ : Membership.mem (Subsemigroup.closure (Set.iUnion fun i => ↑(S i))) x₁
    mem : ∀ (i : ι) (x₂ : M), Membership.mem (S i) x₂ → C x₂
    mul : ∀ (x y : M), C x → C y → C (HMul.hMul x y)
    x₂ : M
    hx₂ : Membership.mem (Set.iUnion fun i => ↑(S i)) x₂
    ⊢ C x₂
  -/
  obtain ⟨i, hi⟩ := Set.mem_iUnion.mp hx₂
  /-
    case intro
    ι : Sort u_1
    M : Type u_2
    inst✝ : Mul M
    S : ι → Subsemigroup M
    C : M → Prop
    x₁ : M
    hx₁ : Membership.mem (Subsemigroup.closure (Set.iUnion fun i => ↑(S i))) x₁
    mem : ∀ (i : ι) (x₂ : M), Membership.mem (S i) x₂ → C x₂
    mul : ∀ (x y : M), C x → C y → C (HMul.hMul x y)
    x₂ : M
    hx₂ : Membership.mem (Set.iUnion fun i => ↑(S i)) x₂
    i : ι
    hi : Membership.mem (↑(S i)) x₂
    ⊢ C x₂
  -/
  exact mem _ _ hi
  /-
    🎉 no goals
  -/


/-- A dependent version of `Subsemigroup.iSup_induction`. -/
@[to_additive (attr := elab_as_elim)
"A dependent version of `AddSubsemigroup.iSup_induction`."]
theorem iSup_induction' (S : ι → Subsemigroup M) {C : ∀ x, (x ∈ ⨆ i, S i) → Prop}
    (mem : ∀ (i) (x) (hxS : x ∈ S i), C x (mem_iSup_of_mem i ‹_›))
    (mul : ∀ x y hx hy, C x hx → C y hy → C (x * y) (mul_mem ‹_› ‹_›)) {x₁ : M}
    (hx₁ : x₁ ∈ ⨆ i, S i) : C x₁ hx₁ := by
  /-
    ι : Sort u_1
    M : Type u_2
    inst✝ : Mul M
    S : ι → Subsemigroup M
    C : (x : M) → Membership.mem (iSup fun i => S i) x → Prop
    mem : ∀ (i : ι) (x : M) (hxS : Membership.mem (S i) x), C x ⋯
    mul : ∀ (x y : M) (hx : Membership.mem (iSup fun i => S i) x) (hy : Membership …
    x₁ : M
    hx₁ : Membership.mem (iSup fun i => S i) x₁
    ⊢ C x₁ hx₁
  -/
  refine Exists.elim ?_ fun (hx₁' : x₁ ∈ ⨆ i, S i) (hc : C x₁ hx₁') => hc
  refine @iSup_induction _ _ _ S (fun x' => ∃ hx'', C x' hx'') _ hx₁
      (fun i x₂ hx₂ => ?_) fun x₃ y => ?_
    /-
      case refine_1
      ι : Sort u_1
      M : Type u_2
      inst✝ : Mul M
      S : ι → Subsemigroup M
      C : (x : M) → Membership.mem (iSup fun i => S i) x → Prop
      mem : ∀ (i : ι) (x : M) (hxS : Membership.mem (S i) x), C x ⋯
      mul : ∀ (x y : M) (hx : Membership.mem (iSup fun i => S i) x) (hy : Membership …
      x₁ : M
      hx₁ : Membership.mem (iSup fun i => S i) x₁
      i : ι
      x₂ : M
      hx₂ : Membership.mem (S i) x₂
      ⊢ (fun x' => Exists fun hx'' => C x' hx'') x₂
    -/
  · exact ⟨_, mem _ _ hx₂⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Sort u_1
      M : Type u_2
      inst✝ : Mul M
      S : ι → Subsemigroup M
      C : (x : M) → Membership.mem (iSup fun i => S i) x → Prop
      mem : ∀ (i : ι) (x : M) (hxS : Membership.mem (S i) x), C x ⋯
      mul : ∀ (x y : M) (hx : Membership.mem (iSup fun i => S i) x) (hy : Membership …
      x₁ : M
      hx₁ : Membership.mem (iSup fun i => S i) x₁
      x₃ y : M
      ⊢ (fun x' => Exists fun hx'' => C x' hx'') x₃ → (fun x' => Exists fun hx'' =>  …
    -/
  · rintro ⟨_, Cx⟩ ⟨_, Cy⟩
    /-
      case refine_2.intro.intro
      ι : Sort u_1
      M : Type u_2
      inst✝ : Mul M
      S : ι → Subsemigroup M
      C : (x : M) → Membership.mem (iSup fun i => S i) x → Prop
      mem : ∀ (i : ι) (x : M) (hxS : Membership.mem (S i) x), C x ⋯
      mul : ∀ (x y : M) (hx : Membership.mem (iSup fun i => S i) x) (hy : Membership …
      x₁ : M
      hx₁ : Membership.mem (iSup fun i => S i) x₁
      x₃ y : M
      w✝¹ : Membership.mem (iSup fun i => S i) x₃
      Cx : C x₃ w✝¹
      w✝ : Membership.mem (iSup fun i => S i) y
      Cy : C y w✝
      ⊢ Exists fun hx'' => C (HMul.hMul x₃ y) hx''
    -/
    exact ⟨_, mul _ _ _ _ Cx Cy⟩
    /-
      🎉 no goals
    -/



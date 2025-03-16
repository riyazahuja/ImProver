/-- Given an endomorphism, `f` of some module, this is the sublattice of all `f`-invariant
submodules. -/
def invtSubmodule : Sublattice (Submodule R M) where
  carrier := {p : Submodule R M | p ≤ p.comap f}
  supClosed' p hp q hq := sup_le_iff.mpr
    ⟨le_trans hp <| Submodule.comap_mono le_sup_left,
    le_trans hq <| Submodule.comap_mono le_sup_right⟩
  infClosed' p hp q hq := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      f : Module.End R M
      p : Submodule R M
      hp : Membership.mem (setOf fun p => LE.le p (Submodule.comap f p)) p
      q : Submodule R M
      hq : Membership.mem (setOf fun p => LE.le p (Submodule.comap f p)) q
      ⊢ Membership.mem (setOf fun p => LE.le p (Submodule.comap f p)) (Min.min p q)
    -/
    simp only [Set.mem_setOf_eq, Submodule.comap_inf, le_inf_iff]
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      f : Module.End R M
      p : Submodule R M
      hp : Membership.mem (setOf fun p => LE.le p (Submodule.comap f p)) p
      q : Submodule R M
      hq : Membership.mem (setOf fun p => LE.le p (Submodule.comap f p)) q
      ⊢ And (LE.le (Min.min p q) (Submodule.comap f p)) (LE.le (Min.min p q) (Submod …
    -/
    exact ⟨inf_le_of_left_le hp, inf_le_of_right_le hq⟩
    /-
      🎉 no goals
    -/


lemma mem_invtSubmodule {p : Submodule R M} :
    p ∈ f.invtSubmodule ↔ p ≤ p.comap f :=
  Iff.rfl


lemma inf_mem {p q : Submodule R M} (hp : p ∈ f.invtSubmodule) (hq : q ∈ f.invtSubmodule) :
    p ⊓ q ∈ f.invtSubmodule :=
  ((⟨p, hp⟩ : f.invtSubmodule) ⊓ (⟨q, hq⟩ : f.invtSubmodule)).property


lemma sup_mem {p q : Submodule R M} (hp : p ∈ f.invtSubmodule) (hq : q ∈ f.invtSubmodule) :
    p ⊔ q ∈ f.invtSubmodule :=
  ((⟨p, hp⟩ : f.invtSubmodule) ⊔ (⟨q, hq⟩ : f.invtSubmodule)).property


@[simp]
                                                    /-
                                                      R : Type u_1
                                                      M : Type u_2
                                                      inst✝² : Semiring R
                                                      inst✝¹ : AddCommMonoid M
                                                      inst✝ : Module R M
                                                      f : Module.End R M
                                                      ⊢ Membership.mem f.invtSubmodule Top.top
                                                    -/
protected lemma top_mem : ⊤ ∈ f.invtSubmodule := by simp [invtSubmodule]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
                                                    /-
                                                      R : Type u_1
                                                      M : Type u_2
                                                      inst✝² : Semiring R
                                                      inst✝¹ : AddCommMonoid M
                                                      inst✝ : Module R M
                                                      f : Module.End R M
                                                      ⊢ Membership.mem f.invtSubmodule Bot.bot
                                                    -/
protected lemma bot_mem : ⊥ ∈ f.invtSubmodule := by simp [invtSubmodule]
                                                    /-
                                                      🎉 no goals
                                                    -/


instance : BoundedOrder (f.invtSubmodule) where
  top := ⟨⊤, invtSubmodule.top_mem f⟩
  bot := ⟨⊥, invtSubmodule.bot_mem f⟩
                             /-
                               R : Type u_1
                               M : Type u_2
                               inst✝² : Semiring R
                               inst✝¹ : AddCommMonoid M
                               inst✝ : Module R M
                               f : Module.End R M
                               x✝ : Subtype fun x => Membership.mem f.invtSubmodule x
                               p : Submodule R M
                               hp : Membership.mem f.invtSubmodule p
                               ⊢ LE.le ⟨p, hp⟩ Top.top
                             -/
  le_top := fun ⟨p, hp⟩ ↦ by simp
                             /-
                               🎉 no goals
                             -/
                             /-
                               R : Type u_1
                               M : Type u_2
                               inst✝² : Semiring R
                               inst✝¹ : AddCommMonoid M
                               inst✝ : Module R M
                               f : Module.End R M
                               x✝ : Subtype fun x => Membership.mem f.invtSubmodule x
                               p : Submodule R M
                               hp : Membership.mem f.invtSubmodule p
                               ⊢ LE.le Bot.bot ⟨p, hp⟩
                             -/
  bot_le := fun ⟨p, hp⟩ ↦ by simp
                             /-
                               🎉 no goals
                             -/


@[simp]
protected lemma zero :
    (0 : End R M).invtSubmodule = ⊤ :=
                            /-
                              R : Type u_1
                              M : Type u_2
                              inst✝² : Semiring R
                              inst✝¹ : AddCommMonoid M
                              inst✝ : Module R M
                              x : Submodule R M
                              ⊢ Membership.mem Top.top x → Membership.mem (Module.End.invtSubmodule 0) x
                            -/
  eq_top_iff.mpr fun x ↦ by simp [invtSubmodule]
                            /-
                              🎉 no goals
                            -/


@[simp]
protected lemma id :
    invtSubmodule (LinearMap.id : End R M) = ⊤ :=
                            /-
                              R : Type u_1
                              M : Type u_2
                              inst✝² : Semiring R
                              inst✝¹ : AddCommMonoid M
                              inst✝ : Module R M
                              x : Submodule R M
                              ⊢ Membership.mem Top.top x → Membership.mem (Module.End.invtSubmodule LinearMa …
                            -/
  eq_top_iff.mpr fun x ↦ by simp [invtSubmodule]
                            /-
                              🎉 no goals
                            -/


protected lemma mk_eq_bot_iff {p : Submodule R M} (hp : p ∈ f.invtSubmodule) :
    (⟨p, hp⟩ : f.invtSubmodule) = ⊥ ↔ p = ⊥ :=
                            /-
                              R : Type u_1
                              M : Type u_2
                              inst✝² : Semiring R
                              inst✝¹ : AddCommMonoid M
                              inst✝ : Module R M
                              f : Module.End R M
                              p : Submodule R M
                              hp : Membership.mem f.invtSubmodule p
                              ⊢ Membership.mem f.invtSubmodule Bot.bot
                            -/
  Subtype.mk_eq_bot_iff (by simp [invtSubmodule]) _
                            /-
                              🎉 no goals
                            -/


protected lemma mk_eq_top_iff {p : Submodule R M} (hp : p ∈ f.invtSubmodule) :
    (⟨p, hp⟩ : f.invtSubmodule) = ⊤ ↔ p = ⊤ :=
                            /-
                              R : Type u_1
                              M : Type u_2
                              inst✝² : Semiring R
                              inst✝¹ : AddCommMonoid M
                              inst✝ : Module R M
                              f : Module.End R M
                              p : Submodule R M
                              hp : Membership.mem f.invtSubmodule p
                              ⊢ Membership.mem f.invtSubmodule Top.top
                            -/
  Subtype.mk_eq_top_iff (by simp [invtSubmodule]) _
                            /-
                              🎉 no goals
                            -/


@[simp]
protected lemma disjoint_mk_iff {p q : Submodule R M}
    (hp : p ∈ f.invtSubmodule) (hq : q ∈ f.invtSubmodule) :
    Disjoint (α := f.invtSubmodule) ⟨p, hp⟩ ⟨q, hq⟩ ↔ Disjoint p q := by
  rw [disjoint_iff, disjoint_iff, Sublattice.mk_inf_mk,
    Subtype.mk_eq_bot_iff (⊥ : f.invtSubmodule).property]


protected lemma disjoint_iff {p q : f.invtSubmodule} :
    Disjoint p q ↔ Disjoint (p : Submodule R M) (q : Submodule R M) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    p q : Subtype fun x => Membership.mem f.invtSubmodule x
    ⊢ Iff (Disjoint p q) (Disjoint ↑p ↑q)
  -/
  obtain ⟨p, hp⟩ := p
  /-
    case mk
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    q : Subtype fun x => Membership.mem f.invtSubmodule x
    p : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    ⊢ Iff (Disjoint ⟨p, hp⟩ q) (Disjoint ↑⟨p, hp⟩ ↑q)
  -/
  obtain ⟨q, hq⟩ := q
  /-
    case mk.mk
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    q : Submodule R M
    hq : Membership.mem f.invtSubmodule q
    ⊢ Iff (Disjoint ⟨p, hp⟩ ⟨q, hq⟩) (Disjoint ↑⟨p, hp⟩ ↑⟨q, hq⟩)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
protected lemma codisjoint_mk_iff {p q : Submodule R M}
    (hp : p ∈ f.invtSubmodule) (hq : q ∈ f.invtSubmodule) :
    Codisjoint (α := f.invtSubmodule) ⟨p, hp⟩ ⟨q, hq⟩ ↔ Codisjoint p q := by
  rw [codisjoint_iff, codisjoint_iff, Sublattice.mk_sup_mk,
    Subtype.mk_eq_top_iff (⊤ : f.invtSubmodule).property]


protected lemma codisjoint_iff {p q : f.invtSubmodule} :
    Codisjoint p q ↔ Codisjoint (p : Submodule R M) (q : Submodule R M) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    p q : Subtype fun x => Membership.mem f.invtSubmodule x
    ⊢ Iff (Codisjoint p q) (Codisjoint ↑p ↑q)
  -/
  obtain ⟨p, hp⟩ := p
  /-
    case mk
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    q : Subtype fun x => Membership.mem f.invtSubmodule x
    p : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    ⊢ Iff (Codisjoint ⟨p, hp⟩ q) (Codisjoint ↑⟨p, hp⟩ ↑q)
  -/
  obtain ⟨q, hq⟩ := q
  /-
    case mk.mk
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    q : Submodule R M
    hq : Membership.mem f.invtSubmodule q
    ⊢ Iff (Codisjoint ⟨p, hp⟩ ⟨q, hq⟩) (Codisjoint ↑⟨p, hp⟩ ↑⟨q, hq⟩)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
protected lemma isCompl_mk_iff {p q : Submodule R M}
    (hp : p ∈ f.invtSubmodule) (hq : q ∈ f.invtSubmodule) :
    IsCompl (α := f.invtSubmodule) ⟨p, hp⟩ ⟨q, hq⟩ ↔ IsCompl p q := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    p q : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    hq : Membership.mem f.invtSubmodule q
    ⊢ Iff (IsCompl ⟨p, hp⟩ ⟨q, hq⟩) (IsCompl p q)
  -/
  simp [isCompl_iff]
  /-
    🎉 no goals
  -/


protected lemma isCompl_iff {p q : f.invtSubmodule} :
    IsCompl p q ↔ IsCompl (p : Submodule R M) (q : Submodule R M) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    p q : Subtype fun x => Membership.mem f.invtSubmodule x
    ⊢ Iff (IsCompl p q) (IsCompl ↑p ↑q)
  -/
  obtain ⟨p, hp⟩ := p
  /-
    case mk
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    q : Subtype fun x => Membership.mem f.invtSubmodule x
    p : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    ⊢ Iff (IsCompl ⟨p, hp⟩ q) (IsCompl ↑⟨p, hp⟩ ↑q)
  -/
  obtain ⟨q, hq⟩ := q
  /-
    case mk.mk
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    q : Submodule R M
    hq : Membership.mem f.invtSubmodule q
    ⊢ Iff (IsCompl ⟨p, hp⟩ ⟨q, hq⟩) (IsCompl ↑⟨p, hp⟩ ↑⟨q, hq⟩)
  -/
  simp
  /-
    🎉 no goals
  -/


lemma map_subtype_mem_of_mem_invtSubmodule {p : Submodule R M} (hp : p ∈ f.invtSubmodule)
    {q : Submodule R p} (hq : q ∈ invtSubmodule (LinearMap.restrict f hp)) :
    Submodule.map p.subtype q ∈ f.invtSubmodule := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    q : Submodule R (Subtype fun x => Membership.mem p x)
    hq : Membership.mem (Module.End.invtSubmodule (LinearMap.restrict f hp)) q
    ⊢ Membership.mem f.invtSubmodule (Submodule.map p.subtype q)
  -/
  rintro - ⟨⟨x, hx⟩, hx', rfl⟩
  /-
    case intro.mk.intro
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    q : Submodule R (Subtype fun x => Membership.mem p x)
    hq : Membership.mem (Module.End.invtSubmodule (LinearMap.restrict f hp)) q
    x : M
    hx : Membership.mem p x
    hx' : Membership.mem ↑q ⟨x, hx⟩
    ⊢ Membership.mem (Submodule.comap f (Submodule.map p.subtype q)) (p.subtype ⟨x …
  -/
  specialize hq hx'
  /-
    case intro.mk.intro
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    q : Submodule R (Subtype fun x => Membership.mem p x)
    x : M
    hx : Membership.mem p x
    hx' : Membership.mem ↑q ⟨x, hx⟩
    hq : Membership.mem (Submodule.comap (LinearMap.restrict f hp) q) ⟨x, hx⟩
    ⊢ Membership.mem (Submodule.comap f (Submodule.map p.subtype q)) (p.subtype ⟨x …
  -/
  rw [Submodule.mem_comap, LinearMap.restrict_apply] at hq
  /-
    case intro.mk.intro
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.End R M
    p : Submodule R M
    hp : Membership.mem f.invtSubmodule p
    q : Submodule R (Subtype fun x => Membership.mem p x)
    x : M
    hx : Membership.mem p x
    hx' : Membership.mem ↑q ⟨x, hx⟩
    hq : Membership.mem q ⟨f ↑⟨x, hx⟩, ⋯⟩
    ⊢ Membership.mem (Submodule.comap f (Submodule.map p.subtype q)) (p.subtype ⟨x …
  -/
  simpa [hq] using hp hx
  /-
    🎉 no goals
  -/



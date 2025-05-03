/-- A generalized Boolean algebra is a distributive lattice with `⊥` and a relative complement
operation `\` (called `sdiff`, after "set difference") satisfying `(a ⊓ b) ⊔ (a \ b) = a` and
`(a ⊓ b) ⊓ (a \ b) = ⊥`, i.e. `a \ b` is the complement of `b` in `a`.

This is a generalization of Boolean algebras which applies to `Finset α` for arbitrary
(not-necessarily-`Fintype`) `α`. -/
class GeneralizedBooleanAlgebra (α : Type u) extends DistribLattice α, SDiff α, Bot α where
  /-- For any `a`, `b`, `(a ⊓ b) ⊔ (a / b) = a` -/
  sup_inf_sdiff : ∀ a b : α, a ⊓ b ⊔ a \ b = a
  /-- For any `a`, `b`, `(a ⊓ b) ⊓ (a / b) = ⊥` -/
  inf_inf_sdiff : ∀ a b : α, a ⊓ b ⊓ a \ b = ⊥

-- We might want an `IsCompl_of` predicate (for relative complements) generalizing `IsCompl`,
-- however we'd need another type class for lattices with bot, and all the API for that.

@[simp]
theorem sup_inf_sdiff (x y : α) : x ⊓ y ⊔ x \ y = x :=
  GeneralizedBooleanAlgebra.sup_inf_sdiff _ _


@[simp]
theorem inf_inf_sdiff (x y : α) : x ⊓ y ⊓ x \ y = ⊥ :=
  GeneralizedBooleanAlgebra.inf_inf_sdiff _ _


@[simp]
                                                          /-
                                                            α : Type u
                                                            inst✝ : GeneralizedBooleanAlgebra α
                                                            x y : α
                                                            ⊢ Eq (Max.max (SDiff.sdiff x y) (Min.min x y)) x
                                                          -/
theorem sup_sdiff_inf (x y : α) : x \ y ⊔ x ⊓ y = x := by rw [sup_comm, sup_inf_sdiff]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
                                                            /-
                                                              α : Type u
                                                              inst✝ : GeneralizedBooleanAlgebra α
                                                              x y : α
                                                              ⊢ Eq (Min.min (SDiff.sdiff x y) (Min.min x y)) Bot.bot
                                                            -/
theorem inf_sdiff_inf (x y : α) : x \ y ⊓ (x ⊓ y) = ⊥ := by rw [inf_comm, inf_inf_sdiff]
                                                            /-
                                                              🎉 no goals
                                                            -/

-- see Note [lower instance priority]

instance (priority := 100) GeneralizedBooleanAlgebra.toOrderBot : OrderBot α where
  __ := GeneralizedBooleanAlgebra.toBot
  bot_le a := by
    /-
      α : Type u
      β : Type u_1
      x y z : α
      inst✝ : GeneralizedBooleanAlgebra α
      a : α
      ⊢ LE.le Bot.bot a
    -/
    rw [← inf_inf_sdiff a a, inf_assoc]
    /-
      α : Type u
      β : Type u_1
      x y z : α
      inst✝ : GeneralizedBooleanAlgebra α
      a : α
      ⊢ LE.le (Min.min a (Min.min a (SDiff.sdiff a a))) a
    -/
    exact inf_le_left
    /-
      🎉 no goals
    -/


theorem disjoint_inf_sdiff : Disjoint (x ⊓ y) (x \ y) :=
  disjoint_iff_inf_le.mpr (inf_inf_sdiff x y).le

-- TODO: in distributive lattices, relative complements are unique when they exist

theorem sdiff_unique (s : x ⊓ y ⊔ z = x) (i : x ⊓ y ⊓ z = ⊥) : x \ y = z := by
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    s : Eq (Max.max (Min.min x y) z) x
    i : Eq (Min.min (Min.min x y) z) Bot.bot
    ⊢ Eq (SDiff.sdiff x y) z
  -/
  conv_rhs at s => rw [← sup_inf_sdiff x y, sup_comm]
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    s : Eq (Max.max (Min.min x y) z) (Max.max (SDiff.sdiff x y) (Min.min x y))
    i : Eq (Min.min (Min.min x y) z) Bot.bot
    ⊢ Eq (SDiff.sdiff x y) z
  -/
  rw [sup_comm] at s
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    s : Eq (Max.max z (Min.min x y)) (Max.max (SDiff.sdiff x y) (Min.min x y))
    i : Eq (Min.min (Min.min x y) z) Bot.bot
    ⊢ Eq (SDiff.sdiff x y) z
  -/
  conv_rhs at i => rw [← inf_inf_sdiff x y, inf_comm]
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    s : Eq (Max.max z (Min.min x y)) (Max.max (SDiff.sdiff x y) (Min.min x y))
    i : Eq (Min.min (Min.min x y) z) (Min.min (SDiff.sdiff x y) (Min.min x y))
    ⊢ Eq (SDiff.sdiff x y) z
  -/
  rw [inf_comm] at i
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    s : Eq (Max.max z (Min.min x y)) (Max.max (SDiff.sdiff x y) (Min.min x y))
    i : Eq (Min.min z (Min.min x y)) (Min.min (SDiff.sdiff x y) (Min.min x y))
    ⊢ Eq (SDiff.sdiff x y) z
  -/
  exact (eq_of_inf_eq_sup_eq i s).symm
  /-
    🎉 no goals
  -/

-- Use `sdiff_le`

private theorem sdiff_le' : x \ y ≤ x :=
  calc
    x \ y ≤ x ⊓ y ⊔ x \ y := le_sup_right
    _ = x := sup_inf_sdiff x y

-- Use `sdiff_sup_self`

private theorem sdiff_sup_self' : y \ x ⊔ x = y ⊔ x :=
  calc
                                          /-
                                            α : Type u
                                            x y : α
                                            inst✝ : GeneralizedBooleanAlgebra α
                                            ⊢ Eq (Max.max (SDiff.sdiff y x) x) (Max.max (SDiff.sdiff y x) (Max.max x (Min. …
                                          -/
    y \ x ⊔ x = y \ x ⊔ (x ⊔ x ⊓ y) := by rw [sup_inf_self]
                                          /-
                                            🎉 no goals
                                          -/
                                /-
                                  α : Type u
                                  x y : α
                                  inst✝ : GeneralizedBooleanAlgebra α
                                  ⊢ Eq (Max.max (SDiff.sdiff y x) (Max.max x (Min.min x y))) (Max.max (Max.max ( …
                                -/
    _ = y ⊓ x ⊔ y \ x ⊔ x := by ac_rfl
                                /-
                                  🎉 no goals
                                -/
                    /-
                      α : Type u
                      x y : α
                      inst✝ : GeneralizedBooleanAlgebra α
                      ⊢ Eq (Max.max (Max.max (Min.min y x) (SDiff.sdiff y x)) x) (Max.max y x)
                    -/
    _ = y ⊔ x := by rw [sup_inf_sdiff]
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem sdiff_inf_sdiff : x \ y ⊓ y \ x = ⊥ :=
  Eq.symm <|
    calc
                              /-
                                α : Type u
                                x y : α
                                inst✝ : GeneralizedBooleanAlgebra α
                                ⊢ Eq Bot.bot (Min.min (Min.min x y) (SDiff.sdiff x y))
                              -/
      ⊥ = x ⊓ y ⊓ x \ y := by rw [inf_inf_sdiff]
                              /-
                                🎉 no goals
                              -/
                                            /-
                                              α : Type u
                                              x y : α
                                              inst✝ : GeneralizedBooleanAlgebra α
                                              ⊢ Eq (Min.min (Min.min x y) (SDiff.sdiff x y)) (Min.min (Min.min x (Max.max (M …
                                            -/
      _ = x ⊓ (y ⊓ x ⊔ y \ x) ⊓ x \ y := by rw [sup_inf_sdiff]
                                            /-
                                              🎉 no goals
                                            -/
                                                  /-
                                                    α : Type u
                                                    x y : α
                                                    inst✝ : GeneralizedBooleanAlgebra α
                                                    ⊢ Eq (Min.min (Min.min x (Max.max (Min.min y x) (SDiff.sdiff y x))) (SDiff.sdi …
                                                  -/
      _ = (x ⊓ (y ⊓ x) ⊔ x ⊓ y \ x) ⊓ x \ y := by rw [inf_sup_left]
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    α : Type u
                                                    x y : α
                                                    inst✝ : GeneralizedBooleanAlgebra α
                                                    ⊢ Eq (Min.min (Max.max (Min.min x (Min.min y x)) (Min.min x (SDiff.sdiff y x)) …
                                                  -/
      _ = (y ⊓ (x ⊓ x) ⊔ x ⊓ y \ x) ⊓ x \ y := by ac_rfl
                                                  /-
                                                    🎉 no goals
                                                  -/
                                            /-
                                              α : Type u
                                              x y : α
                                              inst✝ : GeneralizedBooleanAlgebra α
                                              ⊢ Eq (Min.min (Max.max (Min.min y (Min.min x x)) (Min.min x (SDiff.sdiff y x)) …
                                            -/
      _ = (y ⊓ x ⊔ x ⊓ y \ x) ⊓ x \ y := by rw [inf_idem]
                                            /-
                                              🎉 no goals
                                            -/
                                                  /-
                                                    α : Type u
                                                    x y : α
                                                    inst✝ : GeneralizedBooleanAlgebra α
                                                    ⊢ Eq (Min.min (Max.max (Min.min y x) (Min.min x (SDiff.sdiff y x))) (SDiff.sdi …
                                                  -/
      _ = x ⊓ y ⊓ x \ y ⊔ x ⊓ y \ x ⊓ x \ y := by rw [inf_sup_right, inf_comm x y]
                                                  /-
                                                    🎉 no goals
                                                  -/
                                  /-
                                    α : Type u
                                    x y : α
                                    inst✝ : GeneralizedBooleanAlgebra α
                                    ⊢ Eq (Max.max (Min.min (Min.min x y) (SDiff.sdiff x y)) (Min.min (Min.min x (S …
                                  -/
      _ = x ⊓ y \ x ⊓ x \ y := by rw [inf_inf_sdiff, bot_sup_eq]
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    α : Type u
                                    x y : α
                                    inst✝ : GeneralizedBooleanAlgebra α
                                    ⊢ Eq (Min.min (Min.min x (SDiff.sdiff y x)) (SDiff.sdiff x y)) (Min.min (Min.m …
                                  -/
      _ = x ⊓ x \ y ⊓ y \ x := by ac_rfl
                                  /-
                                    🎉 no goals
                                  -/
                              /-
                                α : Type u
                                x y : α
                                inst✝ : GeneralizedBooleanAlgebra α
                                ⊢ Eq (Min.min (Min.min x (SDiff.sdiff x y)) (SDiff.sdiff y x)) (Min.min (SDiff …
                              -/
      _ = x \ y ⊓ y \ x := by rw [inf_of_le_right sdiff_le']
                              /-
                                🎉 no goals
                              -/


theorem disjoint_sdiff_sdiff : Disjoint (x \ y) (y \ x) :=
  disjoint_iff_inf_le.mpr sdiff_inf_sdiff.le


@[simp]
theorem inf_sdiff_self_right : x ⊓ y \ x = ⊥ :=
  calc
                                              /-
                                                α : Type u
                                                x y : α
                                                inst✝ : GeneralizedBooleanAlgebra α
                                                ⊢ Eq (Min.min x (SDiff.sdiff y x)) (Min.min (Max.max (Min.min x y) (SDiff.sdif …
                                              -/
    x ⊓ y \ x = (x ⊓ y ⊔ x \ y) ⊓ y \ x := by rw [sup_inf_sdiff]
                                              /-
                                                🎉 no goals
                                              -/
                                            /-
                                              α : Type u
                                              x y : α
                                              inst✝ : GeneralizedBooleanAlgebra α
                                              ⊢ Eq (Min.min (Max.max (Min.min x y) (SDiff.sdiff x y)) (SDiff.sdiff y x)) (Ma …
                                            -/
    _ = x ⊓ y ⊓ y \ x ⊔ x \ y ⊓ y \ x := by rw [inf_sup_right]
                                            /-
                                              🎉 no goals
                                            -/
                /-
                  α : Type u
                  x y : α
                  inst✝ : GeneralizedBooleanAlgebra α
                  ⊢ Eq (Max.max (Min.min (Min.min x y) (SDiff.sdiff y x)) (Min.min (SDiff.sdiff  …
                -/
    _ = ⊥ := by rw [inf_comm x y, inf_inf_sdiff, sdiff_inf_sdiff, bot_sup_eq]
                /-
                  🎉 no goals
                -/


@[simp]
                                                  /-
                                                    α : Type u
                                                    x y : α
                                                    inst✝ : GeneralizedBooleanAlgebra α
                                                    ⊢ Eq (Min.min (SDiff.sdiff y x) x) Bot.bot
                                                  -/
theorem inf_sdiff_self_left : y \ x ⊓ x = ⊥ := by rw [inf_comm, inf_sdiff_self_right]
                                                  /-
                                                    🎉 no goals
                                                  -/

-- see Note [lower instance priority]

instance (priority := 100) GeneralizedBooleanAlgebra.toGeneralizedCoheytingAlgebra :
    GeneralizedCoheytingAlgebra α where
  __ := ‹GeneralizedBooleanAlgebra α›
  __ := GeneralizedBooleanAlgebra.toOrderBot
  sdiff := (· \ ·)
  sdiff_le_iff y x z :=
    ⟨fun h =>
      le_of_inf_le_sup_le
        (le_of_eq
          (calc
            y ⊓ y \ x = y \ x := inf_of_le_right sdiff_le'
            _ = x ⊓ y \ x ⊔ z ⊓ y \ x := by
              /-
                α : Type u
                β : Type u_1
                x✝ y✝ z✝ : α
                inst✝ : GeneralizedBooleanAlgebra α
                y x z : α
                h : LE.le (SDiff.sdiff y x) z
                ⊢ Eq (SDiff.sdiff y x) (Max.max (Min.min x (SDiff.sdiff y x)) (Min.min z (SDif …
              -/
              rw [inf_eq_right.2 h, inf_sdiff_self_right, bot_sup_eq]
              /-
                🎉 no goals
              -/
                                      /-
                                        α : Type u
                                        β : Type u_1
                                        x✝ y✝ z✝ : α
                                        inst✝ : GeneralizedBooleanAlgebra α
                                        y x z : α
                                        h : LE.le (SDiff.sdiff y x) z
                                        ⊢ Eq (Max.max (Min.min x (SDiff.sdiff y x)) (Min.min z (SDiff.sdiff y x))) (Mi …
                                      -/
            _ = (x ⊔ z) ⊓ y \ x := by rw [← inf_sup_right]))
                                      /-
                                        🎉 no goals
                                      -/
        (calc
          y ⊔ y \ x = y := sup_of_le_left sdiff_le'
          _ ≤ y ⊔ (x ⊔ z) := le_sup_left
                                  /-
                                    α : Type u
                                    β : Type u_1
                                    x✝ y✝ z✝ : α
                                    inst✝ : GeneralizedBooleanAlgebra α
                                    y x z : α
                                    h : LE.le (SDiff.sdiff y x) z
                                    ⊢ Eq (Max.max y (Max.max x z)) (Max.max (Max.max (SDiff.sdiff y x) x) z)
                                  -/
          _ = y \ x ⊔ x ⊔ z := by rw [← sup_assoc, ← @sdiff_sup_self' _ x y]
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    α : Type u
                                    β : Type u_1
                                    x✝ y✝ z✝ : α
                                    inst✝ : GeneralizedBooleanAlgebra α
                                    y x z : α
                                    h : LE.le (SDiff.sdiff y x) z
                                    ⊢ Eq (Max.max (Max.max (SDiff.sdiff y x) x) z) (Max.max (Max.max x z) (SDiff.s …
                                  -/
          _ = x ⊔ z ⊔ y \ x := by ac_rfl),
                                  /-
                                    🎉 no goals
                                  -/
      fun h =>
      le_of_inf_le_sup_le
        (calc
          y \ x ⊓ x = ⊥ := inf_sdiff_self_left
          _ ≤ z ⊓ x := bot_le)
        (calc
          y \ x ⊔ x = y ⊔ x := sdiff_sup_self'
          _ ≤ x ⊔ z ⊔ x := sup_le_sup_right h x
                          /-
                            α : Type u
                            β : Type u_1
                            x✝ y✝ z✝ : α
                            inst✝ : GeneralizedBooleanAlgebra α
                            y x z : α
                            h : LE.le y (Max.max x z)
                            ⊢ LE.le (Max.max (Max.max x z) x) (Max.max z x)
                          -/
          _ ≤ z ⊔ x := by rw [sup_assoc, sup_comm, sup_assoc, sup_idem])⟩
                          /-
                            🎉 no goals
                          -/


theorem disjoint_sdiff_self_left : Disjoint (y \ x) x :=
  disjoint_iff_inf_le.mpr inf_sdiff_self_left.le


theorem disjoint_sdiff_self_right : Disjoint x (y \ x) :=
  disjoint_iff_inf_le.mpr inf_sdiff_self_right.le


lemma le_sdiff : x ≤ y \ z ↔ x ≤ y ∧ Disjoint x z :=
  ⟨fun h ↦ ⟨h.trans sdiff_le, disjoint_sdiff_self_left.mono_left h⟩, fun h ↦
       /-
         α : Type u
         x y z : α
         inst✝ : GeneralizedBooleanAlgebra α
         h : And (LE.le x y) (Disjoint x z)
         ⊢ LE.le x (SDiff.sdiff y z)
       -/
    by rw [← h.2.sdiff_eq_left]; exact sdiff_le_sdiff_right h.1⟩
                                 /-
                                   🎉 no goals
                                 -/


@[simp] lemma sdiff_eq_left : x \ y = x ↔ Disjoint x y :=
  ⟨fun h ↦ disjoint_sdiff_self_left.mono_left h.ge, Disjoint.sdiff_eq_left⟩

/- TODO: we could make an alternative constructor for `GeneralizedBooleanAlgebra` using
`Disjoint x (y \ x)` and `x ⊔ (y \ x) = y` as axioms. -/

theorem Disjoint.sdiff_eq_of_sup_eq (hi : Disjoint x z) (hs : x ⊔ z = y) : y \ x = z :=
  have h : y ⊓ x = x := inf_eq_right.2 <| le_sup_left.trans hs.le
                   /-
                     α : Type u
                     x y z : α
                     inst✝ : GeneralizedBooleanAlgebra α
                     hi : Disjoint x z
                     hs : Eq (Max.max x z) y
                     h : Eq (Min.min y x) x
                     ⊢ Eq (Max.max (Min.min y x) z) y
                   -/
                   /-
                     🎉 no goals
                   -/
  sdiff_unique (by rw [h, hs]) (by rw [h, hi.eq_bot])
                                   /-
                                     🎉 no goals
                                   -/


protected theorem Disjoint.sdiff_unique (hd : Disjoint x z) (hz : z ≤ y) (hs : y ≤ x ⊔ z) :
    y \ x = z :=
  sdiff_unique
    (by
      /-
        α : Type u
        x y z : α
        inst✝ : GeneralizedBooleanAlgebra α
        hd : Disjoint x z
        hz : LE.le z y
        hs : LE.le y (Max.max x z)
        ⊢ Eq (Max.max (Min.min y x) z) y
      -/
      rw [← inf_eq_right] at hs
      rwa [sup_inf_right, inf_sup_right, sup_comm x, inf_sup_self, inf_comm, sup_comm z,
        hs, sup_eq_left])
        /-
          α : Type u
          x y z : α
          inst✝ : GeneralizedBooleanAlgebra α
          hd : Disjoint x z
          hz : LE.le z y
          hs : LE.le y (Max.max x z)
          ⊢ Eq (Min.min (Min.min y x) z) Bot.bot
        -/
    (by rw [inf_assoc, hd.eq_bot, inf_bot_eq])
        /-
          🎉 no goals
        -/

-- cf. `IsCompl.disjoint_left_iff` and `IsCompl.disjoint_right_iff`

theorem disjoint_sdiff_iff_le (hz : z ≤ y) (hx : x ≤ y) : Disjoint z (y \ x) ↔ z ≤ x :=
  ⟨fun H =>
    le_of_inf_le_sup_le (le_trans H.le_bot bot_le)
      (by
        /-
          α : Type u
          x y z : α
          inst✝ : GeneralizedBooleanAlgebra α
          hz : LE.le z y
          hx : LE.le x y
          H : Disjoint z (SDiff.sdiff y x)
          ⊢ LE.le (Max.max z (SDiff.sdiff y x)) (Max.max x (SDiff.sdiff y x))
        -/
        rw [sup_sdiff_cancel_right hx]
        /-
          α : Type u
          x y z : α
          inst✝ : GeneralizedBooleanAlgebra α
          hz : LE.le z y
          hx : LE.le x y
          H : Disjoint z (SDiff.sdiff y x)
          ⊢ LE.le (Max.max z (SDiff.sdiff y x)) y
        -/
        refine le_trans (sup_le_sup_left sdiff_le z) ?_
        /-
          α : Type u
          x y z : α
          inst✝ : GeneralizedBooleanAlgebra α
          hz : LE.le z y
          hx : LE.le x y
          H : Disjoint z (SDiff.sdiff y x)
          ⊢ LE.le (Max.max z y) y
        -/
        rw [sup_eq_right.2 hz]),
        /-
          🎉 no goals
        -/
    fun H => disjoint_sdiff_self_right.mono_left H⟩

-- cf. `IsCompl.le_left_iff` and `IsCompl.le_right_iff`

theorem le_iff_disjoint_sdiff (hz : z ≤ y) (hx : x ≤ y) : z ≤ x ↔ Disjoint z (y \ x) :=
  (disjoint_sdiff_iff_le hz hx).symm

-- cf. `IsCompl.inf_left_eq_bot_iff` and `IsCompl.inf_right_eq_bot_iff`

theorem inf_sdiff_eq_bot_iff (hz : z ≤ y) (hx : x ≤ y) : z ⊓ y \ x = ⊥ ↔ z ≤ x := by
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    hz : LE.le z y
    hx : LE.le x y
    ⊢ Iff (Eq (Min.min z (SDiff.sdiff y x)) Bot.bot) (LE.le z x)
  -/
  rw [← disjoint_iff]
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    hz : LE.le z y
    hx : LE.le x y
    ⊢ Iff (Disjoint z (SDiff.sdiff y x)) (LE.le z x)
  -/
  exact disjoint_sdiff_iff_le hz hx
  /-
    🎉 no goals
  -/

-- cf. `IsCompl.left_le_iff` and `IsCompl.right_le_iff`

theorem le_iff_eq_sup_sdiff (hz : z ≤ y) (hx : x ≤ y) : x ≤ z ↔ y = z ⊔ y \ x :=
  ⟨fun H => by
    /-
      α : Type u
      x y z : α
      inst✝ : GeneralizedBooleanAlgebra α
      hz : LE.le z y
      hx : LE.le x y
      H : LE.le x z
      ⊢ Eq y (Max.max z (SDiff.sdiff y x))
    -/
    apply le_antisymm
      /-
        case a
        α : Type u
        x y z : α
        inst✝ : GeneralizedBooleanAlgebra α
        hz : LE.le z y
        hx : LE.le x y
        H : LE.le x z
        ⊢ LE.le y (Max.max z (SDiff.sdiff y x))
      -/
    · conv_lhs => rw [← sup_inf_sdiff y x]
      /-
        case a
        α : Type u
        x y z : α
        inst✝ : GeneralizedBooleanAlgebra α
        hz : LE.le z y
        hx : LE.le x y
        H : LE.le x z
        ⊢ LE.le (Max.max (Min.min y x) (SDiff.sdiff y x)) (Max.max z (SDiff.sdiff y x))
      -/
      apply sup_le_sup_right
      /-
        case a.h₁
        α : Type u
        x y z : α
        inst✝ : GeneralizedBooleanAlgebra α
        hz : LE.le z y
        hx : LE.le x y
        H : LE.le x z
        ⊢ LE.le (Min.min y x) z
      -/
      rwa [inf_eq_right.2 hx]
      /-
        🎉 no goals
      -/
      /-
        case a
        α : Type u
        x y z : α
        inst✝ : GeneralizedBooleanAlgebra α
        hz : LE.le z y
        hx : LE.le x y
        H : LE.le x z
        ⊢ LE.le (Max.max z (SDiff.sdiff y x)) y
      -/
    · apply le_trans
        /-
          case a.a
          α : Type u
          x y z : α
          inst✝ : GeneralizedBooleanAlgebra α
          hz : LE.le z y
          hx : LE.le x y
          H : LE.le x z
          ⊢ LE.le (Max.max z (SDiff.sdiff y x)) ?a.b✝
        -/
      · apply sup_le_sup_right hz
        /-
          🎉 no goals
        -/
        /-
          case a.a
          α : Type u
          x y z : α
          inst✝ : GeneralizedBooleanAlgebra α
          hz : LE.le z y
          hx : LE.le x y
          H : LE.le x z
          ⊢ LE.le (Max.max y (SDiff.sdiff y x)) y
        -/
      · rw [sup_sdiff_left],
        /-
          🎉 no goals
        -/
    fun H => by
    /-
      α : Type u
      x y z : α
      inst✝ : GeneralizedBooleanAlgebra α
      hz : LE.le z y
      hx : LE.le x y
      H : Eq y (Max.max z (SDiff.sdiff y x))
      ⊢ LE.le x z
    -/
    conv_lhs at H => rw [← sup_sdiff_cancel_right hx]
    /-
      α : Type u
      x y z : α
      inst✝ : GeneralizedBooleanAlgebra α
      hz : LE.le z y
      hx : LE.le x y
      H : Eq (Max.max x (SDiff.sdiff y x)) (Max.max z (SDiff.sdiff y x))
      ⊢ LE.le x z
    -/
    refine le_of_inf_le_sup_le ?_ H.le
    /-
      α : Type u
      x y z : α
      inst✝ : GeneralizedBooleanAlgebra α
      hz : LE.le z y
      hx : LE.le x y
      H : Eq (Max.max x (SDiff.sdiff y x)) (Max.max z (SDiff.sdiff y x))
      ⊢ LE.le (Min.min x (SDiff.sdiff y x)) (Min.min z (SDiff.sdiff y x))
    -/
    rw [inf_sdiff_self_right]
    /-
      α : Type u
      x y z : α
      inst✝ : GeneralizedBooleanAlgebra α
      hz : LE.le z y
      hx : LE.le x y
      H : Eq (Max.max x (SDiff.sdiff y x)) (Max.max z (SDiff.sdiff y x))
      ⊢ LE.le Bot.bot (Min.min z (SDiff.sdiff y x))
    -/
    exact bot_le⟩
    /-
      🎉 no goals
    -/

-- cf. `IsCompl.sup_inf`

theorem sdiff_sup : y \ (x ⊔ z) = y \ x ⊓ y \ z :=
  sdiff_unique
    (calc
      y ⊓ (x ⊔ z) ⊔ y \ x ⊓ y \ z = (y ⊓ (x ⊔ z) ⊔ y \ x) ⊓ (y ⊓ (x ⊔ z) ⊔ y \ z) := by
          /-
            α : Type u
            x y z : α
            inst✝ : GeneralizedBooleanAlgebra α
            ⊢ Eq (Max.max (Min.min y (Max.max x z)) (Min.min (SDiff.sdiff y x) (SDiff.sdif …
          -/
          rw [sup_inf_left]
          /-
            🎉 no goals
          -/
                                                                  /-
                                                                    α : Type u
                                                                    x y z : α
                                                                    inst✝ : GeneralizedBooleanAlgebra α
                                                                    ⊢ Eq (Min.min (Max.max (Min.min y (Max.max x z)) (SDiff.sdiff y x)) (Max.max ( …
                                                                  -/
      _ = (y ⊓ x ⊔ y ⊓ z ⊔ y \ x) ⊓ (y ⊓ x ⊔ y ⊓ z ⊔ y \ z) := by rw [@inf_sup_left _ _ y]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
                                                                      /-
                                                                        α : Type u
                                                                        x y z : α
                                                                        inst✝ : GeneralizedBooleanAlgebra α
                                                                        ⊢ Eq (Min.min (Max.max (Max.max (Min.min y x) (Min.min y z)) (SDiff.sdiff y x) …
                                                                      -/
      _ = (y ⊓ z ⊔ (y ⊓ x ⊔ y \ x)) ⊓ (y ⊓ x ⊔ (y ⊓ z ⊔ y \ z)) := by ac_rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
                                          /-
                                            α : Type u
                                            x y z : α
                                            inst✝ : GeneralizedBooleanAlgebra α
                                            ⊢ Eq (Min.min (Max.max (Min.min y z) (Max.max (Min.min y x) (SDiff.sdiff y x)) …
                                          -/
      _ = (y ⊓ z ⊔ y) ⊓ (y ⊓ x ⊔ y) := by rw [sup_inf_sdiff, sup_inf_sdiff]
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            α : Type u
                                            x y z : α
                                            inst✝ : GeneralizedBooleanAlgebra α
                                            ⊢ Eq (Min.min (Max.max (Min.min y z) y) (Max.max (Min.min y x) y)) (Min.min (M …
                                          -/
      _ = (y ⊔ y ⊓ z) ⊓ (y ⊔ y ⊓ x) := by ac_rfl
                                          /-
                                            🎉 no goals
                                          -/
                  /-
                    α : Type u
                    x y z : α
                    inst✝ : GeneralizedBooleanAlgebra α
                    ⊢ Eq (Min.min (Max.max y (Min.min y z)) (Max.max y (Min.min y x))) y
                  -/
      _ = y := by rw [sup_inf_self, sup_inf_self, inf_idem])
                  /-
                    🎉 no goals
                  -/
    (calc
                                                                              /-
                                                                                α : Type u
                                                                                x y z : α
                                                                                inst✝ : GeneralizedBooleanAlgebra α
                                                                                ⊢ Eq (Min.min (Min.min y (Max.max x z)) (Min.min (SDiff.sdiff y x) (SDiff.sdif …
                                                                              -/
      y ⊓ (x ⊔ z) ⊓ (y \ x ⊓ y \ z) = (y ⊓ x ⊔ y ⊓ z) ⊓ (y \ x ⊓ y \ z) := by rw [inf_sup_left]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
                                                                  /-
                                                                    α : Type u
                                                                    x y z : α
                                                                    inst✝ : GeneralizedBooleanAlgebra α
                                                                    ⊢ Eq (Min.min (Max.max (Min.min y x) (Min.min y z)) (Min.min (SDiff.sdiff y x) …
                                                                  -/
      _ = y ⊓ x ⊓ (y \ x ⊓ y \ z) ⊔ y ⊓ z ⊓ (y \ x ⊓ y \ z) := by rw [inf_sup_right]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
                                                                  /-
                                                                    α : Type u
                                                                    x y z : α
                                                                    inst✝ : GeneralizedBooleanAlgebra α
                                                                    ⊢ Eq (Max.max (Min.min (Min.min y x) (Min.min (SDiff.sdiff y x) (SDiff.sdiff y …
                                                                  -/
      _ = y ⊓ x ⊓ y \ x ⊓ y \ z ⊔ y \ x ⊓ (y \ z ⊓ (y ⊓ z)) := by ac_rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
      _ = ⊥ := by rw [inf_inf_sdiff, bot_inf_eq, bot_sup_eq, inf_comm (y \ z),
                      inf_inf_sdiff, inf_bot_eq])


theorem sdiff_eq_sdiff_iff_inf_eq_inf : y \ x = y \ z ↔ y ⊓ x = y ⊓ z :=
                                                 /-
                                                   α : Type u
                                                   x y z : α
                                                   inst✝ : GeneralizedBooleanAlgebra α
                                                   h : Eq (SDiff.sdiff y x) (SDiff.sdiff y z)
                                                   ⊢ Eq (Min.min (Min.min y x) (SDiff.sdiff y x)) (Min.min (Min.min y z) (SDiff.s …
                                                 -/
  ⟨fun h => eq_of_inf_eq_sup_eq (a := y \ x) (by rw [inf_inf_sdiff, h, inf_inf_sdiff])
                                                 /-
                                                   🎉 no goals
                                                 -/
        /-
          α : Type u
          x y z : α
          inst✝ : GeneralizedBooleanAlgebra α
          h : Eq (SDiff.sdiff y x) (SDiff.sdiff y z)
          ⊢ Eq (Max.max (Min.min y x) (SDiff.sdiff y x)) (Max.max (Min.min y z) (SDiff.s …
        -/
    (by rw [sup_inf_sdiff, h, sup_inf_sdiff]),
        /-
          🎉 no goals
        -/
                /-
                  α : Type u
                  x y z : α
                  inst✝ : GeneralizedBooleanAlgebra α
                  h : Eq (Min.min y x) (Min.min y z)
                  ⊢ Eq (SDiff.sdiff y x) (SDiff.sdiff y z)
                -/
    fun h => by rw [← sdiff_inf_self_right, ← sdiff_inf_self_right z y, inf_comm, h, inf_comm]⟩
                /-
                  🎉 no goals
                -/


theorem sdiff_eq_self_iff_disjoint : x \ y = x ↔ Disjoint y x :=
  calc
                                    /-
                                      α : Type u
                                      x y : α
                                      inst✝ : GeneralizedBooleanAlgebra α
                                      ⊢ Iff (Eq (SDiff.sdiff x y) x) (Eq (SDiff.sdiff x y) (SDiff.sdiff x Bot.bot))
                                    -/
    x \ y = x ↔ x \ y = x \ ⊥ := by rw [sdiff_bot]
                                    /-
                                      🎉 no goals
                                    -/
    _ ↔ x ⊓ y = x ⊓ ⊥ := sdiff_eq_sdiff_iff_inf_eq_inf
                           /-
                             α : Type u
                             x y : α
                             inst✝ : GeneralizedBooleanAlgebra α
                             ⊢ Iff (Eq (Min.min x y) (Min.min x Bot.bot)) (Disjoint y x)
                           -/
    _ ↔ Disjoint y x := by rw [inf_bot_eq, inf_comm, disjoint_iff]
                           /-
                             🎉 no goals
                           -/


theorem sdiff_eq_self_iff_disjoint' : x \ y = x ↔ Disjoint x y := by
  /-
    α : Type u
    x y : α
    inst✝ : GeneralizedBooleanAlgebra α
    ⊢ Iff (Eq (SDiff.sdiff x y) x) (Disjoint x y)
  -/
  rw [sdiff_eq_self_iff_disjoint, disjoint_comm]
  /-
    🎉 no goals
  -/


theorem sdiff_lt (hx : y ≤ x) (hy : y ≠ ⊥) : x \ y < x := by
  /-
    α : Type u
    x y : α
    inst✝ : GeneralizedBooleanAlgebra α
    hx : LE.le y x
    hy : Ne y Bot.bot
    ⊢ LT.lt (SDiff.sdiff x y) x
  -/
  refine sdiff_le.lt_of_ne fun h => hy ?_
  /-
    α : Type u
    x y : α
    inst✝ : GeneralizedBooleanAlgebra α
    hx : LE.le y x
    hy : Ne y Bot.bot
    h : Eq (SDiff.sdiff x y) x
    ⊢ Eq y Bot.bot
  -/
  rw [sdiff_eq_self_iff_disjoint', disjoint_iff] at h
  /-
    α : Type u
    x y : α
    inst✝ : GeneralizedBooleanAlgebra α
    hx : LE.le y x
    hy : Ne y Bot.bot
    h : Eq (Min.min x y) Bot.bot
    ⊢ Eq y Bot.bot
  -/
  rw [← h, inf_eq_right.mpr hx]
  /-
    🎉 no goals
  -/


@[simp]
theorem le_sdiff_iff : x ≤ y \ x ↔ x = ⊥ :=
  ⟨fun h => disjoint_self.1 (disjoint_sdiff_self_right.mono_right h), fun h => h.le.trans bot_le⟩


@[simp] lemma sdiff_eq_right : x \ y = y ↔ x = ⊥ ∧ y = ⊥ := by
  /-
    α : Type u
    x y : α
    inst✝ : GeneralizedBooleanAlgebra α
    ⊢ Iff (Eq (SDiff.sdiff x y) y) (And (Eq x Bot.bot) (Eq y Bot.bot))
  -/
  rw [disjoint_sdiff_self_left.eq_iff]; aesop
                                        /-
                                          🎉 no goals
                                        -/


lemma sdiff_ne_right : x \ y ≠ y ↔ x ≠ ⊥ ∨ y ≠ ⊥ := sdiff_eq_right.not.trans not_and_or


theorem sdiff_lt_sdiff_right (h : x < y) (hz : z ≤ x) : x \ z < y \ z :=
  (sdiff_le_sdiff_right h.le).lt_of_not_le
    fun h' => h.not_le <| le_sdiff_sup.trans <| sup_le_of_le_sdiff_right h' hz


theorem sup_inf_inf_sdiff : x ⊓ y ⊓ z ⊔ y \ z = x ⊓ y ⊔ y \ z :=
  calc
                                                  /-
                                                    α : Type u
                                                    x y z : α
                                                    inst✝ : GeneralizedBooleanAlgebra α
                                                    ⊢ Eq (Max.max (Min.min (Min.min x y) z) (SDiff.sdiff y z)) (Max.max (Min.min x …
                                                  -/
    x ⊓ y ⊓ z ⊔ y \ z = x ⊓ (y ⊓ z) ⊔ y \ z := by rw [inf_assoc]
                                                  /-
                                                    🎉 no goals
                                                  -/
                              /-
                                α : Type u
                                x y z : α
                                inst✝ : GeneralizedBooleanAlgebra α
                                ⊢ Eq (Max.max (Min.min x (Min.min y z)) (SDiff.sdiff y z)) (Min.min (Max.max x …
                              -/
    _ = (x ⊔ y \ z) ⊓ y := by rw [sup_inf_right, sup_inf_sdiff]
                              /-
                                🎉 no goals
                              -/
                            /-
                              α : Type u
                              x y z : α
                              inst✝ : GeneralizedBooleanAlgebra α
                              ⊢ Eq (Min.min (Max.max x (SDiff.sdiff y z)) y) (Max.max (Min.min x y) (SDiff.s …
                            -/
    _ = x ⊓ y ⊔ y \ z := by rw [inf_sup_right, inf_sdiff_left]
                            /-
                              🎉 no goals
                            -/


theorem sdiff_sdiff_right : x \ (y \ z) = x \ y ⊔ x ⊓ y ⊓ z := by
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    ⊢ Eq (SDiff.sdiff x (SDiff.sdiff y z)) (Max.max (SDiff.sdiff x y) (Min.min (Mi …
  -/
  rw [sup_comm, inf_comm, ← inf_assoc, sup_inf_inf_sdiff]
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    ⊢ Eq (SDiff.sdiff x (SDiff.sdiff y z)) (Max.max (Min.min z x) (SDiff.sdiff x y))
  -/
  apply sdiff_unique
  · calc
      x ⊓ y \ z ⊔ (z ⊓ x ⊔ x \ y) = (x ⊔ (z ⊓ x ⊔ x \ y)) ⊓ (y \ z ⊔ (z ⊓ x ⊔ x \ y)) := by
          rw [sup_inf_right]
      _ = (x ⊔ x ⊓ z ⊔ x \ y) ⊓ (y \ z ⊔ (x ⊓ z ⊔ x \ y)) := by ac_rfl
      _ = x ⊓ (y \ z ⊔ x ⊓ z ⊔ x \ y) := by rw [sup_inf_self, sup_sdiff_left, ← sup_assoc]
      _ = x ⊓ (y \ z ⊓ (z ⊔ y) ⊔ x ⊓ (z ⊔ y) ⊔ x \ y) := by
          rw [sup_inf_left, sdiff_sup_self', inf_sup_right, sup_comm y]
      _ = x ⊓ (y \ z ⊔ (x ⊓ z ⊔ x ⊓ y) ⊔ x \ y) := by
          rw [inf_sdiff_sup_right, @inf_sup_left _ _ x z y]
      _ = x ⊓ (y \ z ⊔ (x ⊓ z ⊔ (x ⊓ y ⊔ x \ y))) := by ac_rfl
      _ = x ⊓ (y \ z ⊔ (x ⊔ x ⊓ z)) := by rw [sup_inf_sdiff, sup_comm (x ⊓ z)]
      _ = x := by rw [sup_inf_self, sup_comm, inf_sup_self]
  · calc
      x ⊓ y \ z ⊓ (z ⊓ x ⊔ x \ y) = x ⊓ y \ z ⊓ (z ⊓ x) ⊔ x ⊓ y \ z ⊓ x \ y := by rw [inf_sup_left]
      _ = x ⊓ (y \ z ⊓ z ⊓ x) ⊔ x ⊓ y \ z ⊓ x \ y := by ac_rfl
      _ = x ⊓ y \ z ⊓ x \ y := by rw [inf_sdiff_self_left, bot_inf_eq, inf_bot_eq, bot_sup_eq]
      _ = x ⊓ (y \ z ⊓ y) ⊓ x \ y := by conv_lhs => rw [← inf_sdiff_left]
      _ = x ⊓ (y \ z ⊓ (y ⊓ x \ y)) := by ac_rfl
      _ = ⊥ := by rw [inf_sdiff_self_right, inf_bot_eq, inf_bot_eq]


theorem sdiff_sdiff_right' : x \ (y \ z) = x \ y ⊔ x ⊓ z :=
  calc
    x \ (y \ z) = x \ y ⊔ x ⊓ y ⊓ z := sdiff_sdiff_right
                                /-
                                  α : Type u
                                  x y z : α
                                  inst✝ : GeneralizedBooleanAlgebra α
                                  ⊢ Eq (Max.max (SDiff.sdiff x y) (Min.min (Min.min x y) z)) (Max.max (Min.min ( …
                                -/
    _ = z ⊓ x ⊓ y ⊔ x \ y := by ac_rfl
                                /-
                                  🎉 no goals
                                -/
                            /-
                              α : Type u
                              x y z : α
                              inst✝ : GeneralizedBooleanAlgebra α
                              ⊢ Eq (Max.max (Min.min (Min.min z x) y) (SDiff.sdiff x y)) (Max.max (SDiff.sdi …
                            -/
    _ = x \ y ⊔ x ⊓ z := by rw [sup_inf_inf_sdiff, sup_comm, inf_comm]
                            /-
                              🎉 no goals
                            -/


theorem sdiff_sdiff_eq_sdiff_sup (h : z ≤ x) : x \ (y \ z) = x \ y ⊔ z := by
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    h : LE.le z x
    ⊢ Eq (SDiff.sdiff x (SDiff.sdiff y z)) (Max.max (SDiff.sdiff x y) z)
  -/
  rw [sdiff_sdiff_right', inf_eq_right.2 h]
  /-
    🎉 no goals
  -/


@[simp]
theorem sdiff_sdiff_right_self : x \ (x \ y) = x ⊓ y := by
  /-
    α : Type u
    x y : α
    inst✝ : GeneralizedBooleanAlgebra α
    ⊢ Eq (SDiff.sdiff x (SDiff.sdiff x y)) (Min.min x y)
  -/
  rw [sdiff_sdiff_right, inf_idem, sdiff_self, bot_sup_eq]
  /-
    🎉 no goals
  -/


theorem sdiff_sdiff_eq_self (h : y ≤ x) : x \ (x \ y) = y := by
  /-
    α : Type u
    x y : α
    inst✝ : GeneralizedBooleanAlgebra α
    h : LE.le y x
    ⊢ Eq (SDiff.sdiff x (SDiff.sdiff x y)) y
  -/
  rw [sdiff_sdiff_right_self, inf_of_le_right h]
  /-
    🎉 no goals
  -/


theorem sdiff_eq_symm (hy : y ≤ x) (h : x \ y = z) : x \ z = y := by
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    hy : LE.le y x
    h : Eq (SDiff.sdiff x y) z
    ⊢ Eq (SDiff.sdiff x z) y
  -/
  rw [← h, sdiff_sdiff_eq_self hy]
  /-
    🎉 no goals
  -/


theorem sdiff_eq_comm (hy : y ≤ x) (hz : z ≤ x) : x \ y = z ↔ x \ z = y :=
  ⟨sdiff_eq_symm hy, sdiff_eq_symm hz⟩


theorem eq_of_sdiff_eq_sdiff (hxz : x ≤ z) (hyz : y ≤ z) (h : z \ x = z \ y) : x = y := by
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    hxz : LE.le x z
    hyz : LE.le y z
    h : Eq (SDiff.sdiff z x) (SDiff.sdiff z y)
    ⊢ Eq x y
  -/
  rw [← sdiff_sdiff_eq_self hxz, h, sdiff_sdiff_eq_self hyz]
  /-
    🎉 no goals
  -/


theorem sdiff_le_sdiff_iff_le (hx : x ≤ z) (hy : y ≤ z) : z \ x ≤ z \ y ↔ y ≤ x := by
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    hx : LE.le x z
    hy : LE.le y z
    ⊢ Iff (LE.le (SDiff.sdiff z x) (SDiff.sdiff z y)) (LE.le y x)
  -/
  refine ⟨fun h ↦ ?_, sdiff_le_sdiff_left⟩
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    hx : LE.le x z
    hy : LE.le y z
    h : LE.le (SDiff.sdiff z x) (SDiff.sdiff z y)
    ⊢ LE.le y x
  -/
  rw [← sdiff_sdiff_eq_self hx, ← sdiff_sdiff_eq_self hy]
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    hx : LE.le x z
    hy : LE.le y z
    h : LE.le (SDiff.sdiff z x) (SDiff.sdiff z y)
    ⊢ LE.le (SDiff.sdiff z (SDiff.sdiff z y)) (SDiff.sdiff z (SDiff.sdiff z x))
  -/
  exact sdiff_le_sdiff_left h
  /-
    🎉 no goals
  -/


                                                              /-
                                                                α : Type u
                                                                x y z : α
                                                                inst✝ : GeneralizedBooleanAlgebra α
                                                                ⊢ Eq (SDiff.sdiff (SDiff.sdiff x y) z) (Min.min (SDiff.sdiff x y) (SDiff.sdiff …
                                                              -/
theorem sdiff_sdiff_left' : (x \ y) \ z = x \ y ⊓ x \ z := by rw [sdiff_sdiff_left, sdiff_sup]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem sdiff_sdiff_sup_sdiff : z \ (x \ y ⊔ y \ x) = z ⊓ (z \ x ⊔ y) ⊓ (z \ y ⊔ x) :=
  calc
    z \ (x \ y ⊔ y \ x) = (z \ x ⊔ z ⊓ x ⊓ y) ⊓ (z \ y ⊔ z ⊓ y ⊓ x) := by
        /-
          α : Type u
          x y z : α
          inst✝ : GeneralizedBooleanAlgebra α
          ⊢ Eq (SDiff.sdiff z (Max.max (SDiff.sdiff x y) (SDiff.sdiff y x))) (Min.min (M …
        -/
        rw [sdiff_sup, sdiff_sdiff_right, sdiff_sdiff_right]
        /-
          🎉 no goals
        -/
                                                    /-
                                                      α : Type u
                                                      x y z : α
                                                      inst✝ : GeneralizedBooleanAlgebra α
                                                      ⊢ Eq (Min.min (Max.max (SDiff.sdiff z x) (Min.min (Min.min z x) y)) (Max.max ( …
                                                    -/
    _ = z ⊓ (z \ x ⊔ y) ⊓ (z \ y ⊔ z ⊓ y ⊓ x) := by rw [sup_inf_left, sup_comm, sup_inf_sdiff]
                                                    /-
                                                      🎉 no goals
                                                    -/
    _ = z ⊓ (z \ x ⊔ y) ⊓ (z ⊓ (z \ y ⊔ x)) := by
        /-
          α : Type u
          x y z : α
          inst✝ : GeneralizedBooleanAlgebra α
          ⊢ Eq (Min.min (Min.min z (Max.max (SDiff.sdiff z x) y)) (Max.max (SDiff.sdiff  …
        -/
        rw [sup_inf_left, sup_comm (z \ y), sup_inf_sdiff]
        /-
          🎉 no goals
        -/
                                                /-
                                                  α : Type u
                                                  x y z : α
                                                  inst✝ : GeneralizedBooleanAlgebra α
                                                  ⊢ Eq (Min.min (Min.min z (Max.max (SDiff.sdiff z x) y)) (Min.min z (Max.max (S …
                                                -/
    _ = z ⊓ z ⊓ (z \ x ⊔ y) ⊓ (z \ y ⊔ x) := by ac_rfl
                                                /-
                                                  🎉 no goals
                                                -/
                                            /-
                                              α : Type u
                                              x y z : α
                                              inst✝ : GeneralizedBooleanAlgebra α
                                              ⊢ Eq (Min.min (Min.min (Min.min z z) (Max.max (SDiff.sdiff z x) y)) (Max.max ( …
                                            -/
    _ = z ⊓ (z \ x ⊔ y) ⊓ (z \ y ⊔ x) := by rw [inf_idem]
                                            /-
                                              🎉 no goals
                                            -/


theorem sdiff_sdiff_sup_sdiff' : z \ (x \ y ⊔ y \ x) = z ⊓ x ⊓ y ⊔ z \ x ⊓ z \ y :=
  calc
    z \ (x \ y ⊔ y \ x) = z \ (x \ y) ⊓ z \ (y \ x) := sdiff_sup
                                                        /-
                                                          α : Type u
                                                          x y z : α
                                                          inst✝ : GeneralizedBooleanAlgebra α
                                                          ⊢ Eq (Min.min (SDiff.sdiff z (SDiff.sdiff x y)) (SDiff.sdiff z (SDiff.sdiff y  …
                                                        -/
    _ = (z \ x ⊔ z ⊓ x ⊓ y) ⊓ (z \ y ⊔ z ⊓ y ⊓ x) := by rw [sdiff_sdiff_right, sdiff_sdiff_right]
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          α : Type u
                                                          x y z : α
                                                          inst✝ : GeneralizedBooleanAlgebra α
                                                          ⊢ Eq (Min.min (Max.max (SDiff.sdiff z x) (Min.min (Min.min z x) y)) (Max.max ( …
                                                        -/
    _ = (z \ x ⊔ z ⊓ y ⊓ x) ⊓ (z \ y ⊔ z ⊓ y ⊓ x) := by ac_rfl
                                                        /-
                                                          🎉 no goals
                                                        -/
                                        /-
                                          α : Type u
                                          x y z : α
                                          inst✝ : GeneralizedBooleanAlgebra α
                                          ⊢ Eq (Min.min (Max.max (SDiff.sdiff z x) (Min.min (Min.min z y) x)) (Max.max ( …
                                        -/
    _ = z \ x ⊓ z \ y ⊔ z ⊓ y ⊓ x := by rw [← sup_inf_right]
                                        /-
                                          🎉 no goals
                                        -/
                                        /-
                                          α : Type u
                                          x y z : α
                                          inst✝ : GeneralizedBooleanAlgebra α
                                          ⊢ Eq (Max.max (Min.min (SDiff.sdiff z x) (SDiff.sdiff z y)) (Min.min (Min.min  …
                                        -/
    _ = z ⊓ x ⊓ y ⊔ z \ x ⊓ z \ y := by ac_rfl
                                        /-
                                          🎉 no goals
                                        -/


lemma sdiff_sdiff_sdiff_cancel_left (hca : z ≤ x) : (x \ y) \ (x \ z) = z \ y :=
  sdiff_sdiff_sdiff_le_sdiff.antisymm <|
    (disjoint_sdiff_self_right.mono_left sdiff_le).le_sdiff_of_le_left <| sdiff_le_sdiff_right hca


lemma sdiff_sdiff_sdiff_cancel_right (hcb : z ≤ y) : (x \ z) \ (y \ z) = x \ y := by
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    hcb : LE.le z y
    ⊢ Eq (SDiff.sdiff (SDiff.sdiff x z) (SDiff.sdiff y z)) (SDiff.sdiff x y)
  -/
  rw [le_antisymm_iff, sdiff_le_comm]
  exact ⟨sdiff_sdiff_sdiff_le_sdiff,
    (disjoint_sdiff_self_left.mono_right sdiff_le).le_sdiff_of_le_left <| sdiff_le_sdiff_left hcb⟩


theorem inf_sdiff : (x ⊓ y) \ z = x \ z ⊓ y \ z :=
  sdiff_unique
    (calc
                                                                                  /-
                                                                                    α : Type u
                                                                                    x y z : α
                                                                                    inst✝ : GeneralizedBooleanAlgebra α
                                                                                    ⊢ Eq (Max.max (Min.min (Min.min x y) z) (Min.min (SDiff.sdiff x z) (SDiff.sdif …
                                                                                  -/
      x ⊓ y ⊓ z ⊔ x \ z ⊓ y \ z = (x ⊓ y ⊓ z ⊔ x \ z) ⊓ (x ⊓ y ⊓ z ⊔ y \ z) := by rw [sup_inf_left]
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
      _ = (x ⊓ y ⊓ (z ⊔ x) ⊔ x \ z) ⊓ (x ⊓ y ⊓ z ⊔ y \ z) := by
          /-
            α : Type u
            x y z : α
            inst✝ : GeneralizedBooleanAlgebra α
            ⊢ Eq (Min.min (Max.max (Min.min (Min.min x y) z) (SDiff.sdiff x z)) (Max.max ( …
          -/
          rw [sup_inf_right, sup_sdiff_self_right, inf_sup_right, inf_sdiff_sup_right]
          /-
            🎉 no goals
          -/
                                                                  /-
                                                                    α : Type u
                                                                    x y z : α
                                                                    inst✝ : GeneralizedBooleanAlgebra α
                                                                    ⊢ Eq (Min.min (Max.max (Min.min (Min.min x y) (Max.max z x)) (SDiff.sdiff x z) …
                                                                  -/
      _ = (y ⊓ (x ⊓ (x ⊔ z)) ⊔ x \ z) ⊓ (x ⊓ y ⊓ z ⊔ y \ z) := by ac_rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
                                                  /-
                                                    α : Type u
                                                    x y z : α
                                                    inst✝ : GeneralizedBooleanAlgebra α
                                                    ⊢ Eq (Min.min (Max.max (Min.min y (Min.min x (Max.max x z))) (SDiff.sdiff x z) …
                                                  -/
      _ = (y ⊓ x ⊔ x \ z) ⊓ (x ⊓ y ⊔ y \ z) := by rw [inf_sup_self, sup_inf_inf_sdiff]
                                                  /-
                                                    🎉 no goals
                                                  -/
                                      /-
                                        α : Type u
                                        x y z : α
                                        inst✝ : GeneralizedBooleanAlgebra α
                                        ⊢ Eq (Min.min (Max.max (Min.min y x) (SDiff.sdiff x z)) (Max.max (Min.min x y) …
                                      -/
      _ = x ⊓ y ⊔ x \ z ⊓ y \ z := by rw [inf_comm y, sup_inf_left]
                                      /-
                                        🎉 no goals
                                      -/
      _ = x ⊓ y := sup_eq_left.2 (inf_le_inf sdiff_le sdiff_le))
    (calc
                                                                      /-
                                                                        α : Type u
                                                                        x y z : α
                                                                        inst✝ : GeneralizedBooleanAlgebra α
                                                                        ⊢ Eq (Min.min (Min.min (Min.min x y) z) (Min.min (SDiff.sdiff x z) (SDiff.sdif …
                                                                      -/
      x ⊓ y ⊓ z ⊓ (x \ z ⊓ y \ z) = x ⊓ y ⊓ (z ⊓ x \ z) ⊓ y \ z := by ac_rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
                  /-
                    α : Type u
                    x y z : α
                    inst✝ : GeneralizedBooleanAlgebra α
                    ⊢ Eq (Min.min (Min.min (Min.min x y) (Min.min z (SDiff.sdiff x z))) (SDiff.sdi …
                  -/
      _ = ⊥ := by rw [inf_sdiff_self_right, inf_bot_eq, bot_inf_eq])
                  /-
                    🎉 no goals
                  -/


theorem inf_sdiff_assoc : (x ⊓ y) \ z = x ⊓ y \ z :=
  sdiff_unique
    (calc
                                                            /-
                                                              α : Type u
                                                              x y z : α
                                                              inst✝ : GeneralizedBooleanAlgebra α
                                                              ⊢ Eq (Max.max (Min.min (Min.min x y) z) (Min.min x (SDiff.sdiff y z))) (Max.ma …
                                                            -/
      x ⊓ y ⊓ z ⊔ x ⊓ y \ z = x ⊓ (y ⊓ z) ⊔ x ⊓ y \ z := by rw [inf_assoc]
                                                            /-
                                                              🎉 no goals
                                                            -/
                                    /-
                                      α : Type u
                                      x y z : α
                                      inst✝ : GeneralizedBooleanAlgebra α
                                      ⊢ Eq (Max.max (Min.min x (Min.min y z)) (Min.min x (SDiff.sdiff y z))) (Min.mi …
                                    -/
      _ = x ⊓ (y ⊓ z ⊔ y \ z) := by rw [← inf_sup_left]
                                    /-
                                      🎉 no goals
                                    -/
                      /-
                        α : Type u
                        x y z : α
                        inst✝ : GeneralizedBooleanAlgebra α
                        ⊢ Eq (Min.min x (Max.max (Min.min y z) (SDiff.sdiff y z))) (Min.min x y)
                      -/
      _ = x ⊓ y := by rw [sup_inf_sdiff])
                      /-
                        🎉 no goals
                      -/
    (calc
                                                              /-
                                                                α : Type u
                                                                x y z : α
                                                                inst✝ : GeneralizedBooleanAlgebra α
                                                                ⊢ Eq (Min.min (Min.min (Min.min x y) z) (Min.min x (SDiff.sdiff y z))) (Min.mi …
                                                              -/
      x ⊓ y ⊓ z ⊓ (x ⊓ y \ z) = x ⊓ x ⊓ (y ⊓ z ⊓ y \ z) := by ac_rfl
                                                              /-
                                                                🎉 no goals
                                                              -/
                  /-
                    α : Type u
                    x y z : α
                    inst✝ : GeneralizedBooleanAlgebra α
                    ⊢ Eq (Min.min (Min.min x x) (Min.min (Min.min y z) (SDiff.sdiff y z))) Bot.bot
                  -/
      _ = ⊥ := by rw [inf_inf_sdiff, inf_bot_eq])
                  /-
                    🎉 no goals
                  -/


theorem inf_sdiff_right_comm : x \ z ⊓ y = (x ⊓ y) \ z := by
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    ⊢ Eq (Min.min (SDiff.sdiff x z) y) (SDiff.sdiff (Min.min x y) z)
  -/
  rw [inf_comm x, inf_comm, inf_sdiff_assoc]
  /-
    🎉 no goals
  -/


theorem inf_sdiff_distrib_left (a b c : α) : a ⊓ b \ c = (a ⊓ b) \ (a ⊓ c) := by
  /-
    α : Type u
    inst✝ : GeneralizedBooleanAlgebra α
    a b c : α
    ⊢ Eq (Min.min a (SDiff.sdiff b c)) (SDiff.sdiff (Min.min a b) (Min.min a c))
  -/
  rw [sdiff_inf, sdiff_eq_bot_iff.2 inf_le_left, bot_sup_eq, inf_sdiff_assoc]
  /-
    🎉 no goals
  -/


theorem inf_sdiff_distrib_right (a b c : α) : a \ b ⊓ c = (a ⊓ c) \ (b ⊓ c) := by
  /-
    α : Type u
    inst✝ : GeneralizedBooleanAlgebra α
    a b c : α
    ⊢ Eq (Min.min (SDiff.sdiff a b) c) (SDiff.sdiff (Min.min a c) (Min.min b c))
  -/
  simp_rw [inf_comm _ c, inf_sdiff_distrib_left]
  /-
    🎉 no goals
  -/


theorem disjoint_sdiff_comm : Disjoint (x \ z) y ↔ Disjoint x (y \ z) := by
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    ⊢ Iff (Disjoint (SDiff.sdiff x z) y) (Disjoint x (SDiff.sdiff y z))
  -/
  simp_rw [disjoint_iff, inf_sdiff_right_comm, inf_sdiff_assoc]
  /-
    🎉 no goals
  -/


theorem sup_eq_sdiff_sup_sdiff_sup_inf : x ⊔ y = x \ y ⊔ y \ x ⊔ x ⊓ y :=
  Eq.symm <|
    calc
                                                                              /-
                                                                                α : Type u
                                                                                x y : α
                                                                                inst✝ : GeneralizedBooleanAlgebra α
                                                                                ⊢ Eq (Max.max (Max.max (SDiff.sdiff x y) (SDiff.sdiff y x)) (Min.min x y)) (Mi …
                                                                              -/
      x \ y ⊔ y \ x ⊔ x ⊓ y = (x \ y ⊔ y \ x ⊔ x) ⊓ (x \ y ⊔ y \ x ⊔ y) := by rw [sup_inf_left]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
                                                            /-
                                                              α : Type u
                                                              x y : α
                                                              inst✝ : GeneralizedBooleanAlgebra α
                                                              ⊢ Eq (Min.min (Max.max (Max.max (SDiff.sdiff x y) (SDiff.sdiff y x)) x) (Max.m …
                                                            -/
      _ = (x \ y ⊔ x ⊔ y \ x) ⊓ (x \ y ⊔ (y \ x ⊔ y)) := by ac_rfl
                                                            /-
                                                              🎉 no goals
                                                            -/
                                          /-
                                            α : Type u
                                            x y : α
                                            inst✝ : GeneralizedBooleanAlgebra α
                                            ⊢ Eq (Min.min (Max.max (Max.max (SDiff.sdiff x y) x) (SDiff.sdiff y x)) (Max.m …
                                          -/
      _ = (x ⊔ y \ x) ⊓ (x \ y ⊔ y) := by rw [sup_sdiff_right, sup_sdiff_right]
                                          /-
                                            🎉 no goals
                                          -/
                      /-
                        α : Type u
                        x y : α
                        inst✝ : GeneralizedBooleanAlgebra α
                        ⊢ Eq (Min.min (Max.max x (SDiff.sdiff y x)) (Max.max (SDiff.sdiff x y) y)) (Ma …
                      -/
      _ = x ⊔ y := by rw [sup_sdiff_self_right, sup_sdiff_self_left, inf_idem]
                      /-
                        🎉 no goals
                      -/


theorem sup_lt_of_lt_sdiff_left (h : y < z \ x) (hxz : x ≤ z) : x ⊔ y < z := by
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    h : LT.lt y (SDiff.sdiff z x)
    hxz : LE.le x z
    ⊢ LT.lt (Max.max x y) z
  -/
  rw [← sup_sdiff_cancel_right hxz]
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    h : LT.lt y (SDiff.sdiff z x)
    hxz : LE.le x z
    ⊢ LT.lt (Max.max x y) (Max.max x (SDiff.sdiff z x))
  -/
  refine (sup_le_sup_left h.le _).lt_of_not_le fun h' => h.not_le ?_
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    h : LT.lt y (SDiff.sdiff z x)
    hxz : LE.le x z
    h' : LE.le (Max.max x (SDiff.sdiff z x)) (Max.max x y)
    ⊢ LE.le (SDiff.sdiff z x) y
  -/
  rw [← sdiff_idem]
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    h : LT.lt y (SDiff.sdiff z x)
    hxz : LE.le x z
    h' : LE.le (Max.max x (SDiff.sdiff z x)) (Max.max x y)
    ⊢ LE.le (SDiff.sdiff (SDiff.sdiff z x) x) y
  -/
  exact (sdiff_le_sdiff_of_sup_le_sup_left h').trans sdiff_le
  /-
    🎉 no goals
  -/


theorem sup_lt_of_lt_sdiff_right (h : x < z \ y) (hyz : y ≤ z) : x ⊔ y < z := by
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    h : LT.lt x (SDiff.sdiff z y)
    hyz : LE.le y z
    ⊢ LT.lt (Max.max x y) z
  -/
  rw [← sdiff_sup_cancel hyz]
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    h : LT.lt x (SDiff.sdiff z y)
    hyz : LE.le y z
    ⊢ LT.lt (Max.max x y) (Max.max (SDiff.sdiff z y) y)
  -/
  refine (sup_le_sup_right h.le _).lt_of_not_le fun h' => h.not_le ?_
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    h : LT.lt x (SDiff.sdiff z y)
    hyz : LE.le y z
    h' : LE.le (Max.max (SDiff.sdiff z y) y) (Max.max x y)
    ⊢ LE.le (SDiff.sdiff z y) x
  -/
  rw [← sdiff_idem]
  /-
    α : Type u
    x y z : α
    inst✝ : GeneralizedBooleanAlgebra α
    h : LT.lt x (SDiff.sdiff z y)
    hyz : LE.le y z
    h' : LE.le (Max.max (SDiff.sdiff z y) y) (Max.max x y)
    ⊢ LE.le (SDiff.sdiff (SDiff.sdiff z y) y) x
  -/
  exact (sdiff_le_sdiff_of_sup_le_sup_right h').trans sdiff_le
  /-
    🎉 no goals
  -/


instance Prod.instGeneralizedBooleanAlgebra [GeneralizedBooleanAlgebra β] :
    GeneralizedBooleanAlgebra (α × β) where
  sup_inf_sdiff _ _ := Prod.ext (sup_inf_sdiff _ _) (sup_inf_sdiff _ _)
  inf_inf_sdiff _ _ := Prod.ext (inf_inf_sdiff _ _) (inf_inf_sdiff _ _)

-- Porting note:
-- Once `pi_instance` has been ported, this is just `by pi_instance`.

instance Pi.instGeneralizedBooleanAlgebra {ι : Type*} {α : ι → Type*}
    [∀ i, GeneralizedBooleanAlgebra (α i)] : GeneralizedBooleanAlgebra (∀ i, α i) where
  sup_inf_sdiff := fun f g => funext fun a => sup_inf_sdiff (f a) (g a)
  inf_inf_sdiff := fun f g => funext fun a => inf_inf_sdiff (f a) (g a)


/-- A Boolean algebra is a bounded distributive lattice with a complement operator `ᶜ` such that
`x ⊓ xᶜ = ⊥` and `x ⊔ xᶜ = ⊤`. For convenience, it must also provide a set difference operation `\`
and a Heyting implication `⇨` satisfying `x \ y = x ⊓ yᶜ` and `x ⇨ y = y ⊔ xᶜ`.

This is a generalization of (classical) logic of propositions, or the powerset lattice.

Since `BoundedOrder`, `OrderBot`, and `OrderTop` are mixins that require `LE`
to be present at define-time, the `extends` mechanism does not work with them.
Instead, we extend using the underlying `Bot` and `Top` data typeclasses, and replicate the
order axioms of those classes here. A "forgetful" instance back to `BoundedOrder` is provided.
-/
class BooleanAlgebra (α : Type u) extends
    DistribLattice α, HasCompl α, SDiff α, HImp α, Top α, Bot α where
  /-- The infimum of `x` and `xᶜ` is at most `⊥` -/
  inf_compl_le_bot : ∀ x : α, x ⊓ xᶜ ≤ ⊥
  /-- The supremum of `x` and `xᶜ` is at least `⊤` -/
  top_le_sup_compl : ∀ x : α, ⊤ ≤ x ⊔ xᶜ
  /-- `⊤` is the greatest element -/
  le_top : ∀ a : α, a ≤ ⊤
  /-- `⊥` is the least element -/
  bot_le : ∀ a : α, ⊥ ≤ a
  /-- `x \ y` is equal to `x ⊓ yᶜ` -/
  sdiff := fun x y => x ⊓ yᶜ
  /-- `x ⇨ y` is equal to `y ⊔ xᶜ` -/
  himp := fun x y => y ⊔ xᶜ
  /-- `x \ y` is equal to `x ⊓ yᶜ` -/
  sdiff_eq : ∀ x y : α, x \ y = x ⊓ yᶜ := by aesop
  /-- `x ⇨ y` is equal to `y ⊔ xᶜ` -/
  himp_eq : ∀ x y : α, x ⇨ y = y ⊔ xᶜ := by aesop

-- see Note [lower instance priority]

instance (priority := 100) BooleanAlgebra.toBoundedOrder [h : BooleanAlgebra α] : BoundedOrder α :=
  { h with }

-- See note [reducible non instances]

/-- A bounded generalized boolean algebra is a boolean algebra. -/
abbrev GeneralizedBooleanAlgebra.toBooleanAlgebra [GeneralizedBooleanAlgebra α] [OrderTop α] :
    BooleanAlgebra α where
  __ := ‹GeneralizedBooleanAlgebra α›
  __ := GeneralizedBooleanAlgebra.toOrderBot
  __ := ‹OrderTop α›
  compl a := ⊤ \ a
  inf_compl_le_bot _ := disjoint_sdiff_self_right.le_bot
  top_le_sup_compl _ := le_sup_sdiff
  sdiff_eq _ _ := by
      -- Porting note: changed `rw` to `erw` here.
      -- https://github.com/leanprover-community/mathlib4/issues/5164
      /-
        α : Type u
        β : Type u_1
        x y z : α
        inst✝¹ : GeneralizedBooleanAlgebra α
        inst✝ : OrderTop α
        x✝¹ x✝ : α
        ⊢ Eq (SDiff.sdiff x✝¹ x✝) (Min.min x✝¹ (HasCompl.compl x✝))
      -/
      erw [← inf_sdiff_assoc, inf_top_eq]
      /-
        🎉 no goals
      -/


theorem inf_compl_eq_bot' : x ⊓ xᶜ = ⊥ :=
  bot_unique <| BooleanAlgebra.inf_compl_le_bot x


@[simp]
theorem sup_compl_eq_top : x ⊔ xᶜ = ⊤ :=
  top_unique <| BooleanAlgebra.top_le_sup_compl x


@[simp]
                                            /-
                                              α : Type u
                                              x : α
                                              inst✝ : BooleanAlgebra α
                                              ⊢ Eq (Max.max (HasCompl.compl x) x) Top.top
                                            -/
theorem compl_sup_eq_top : xᶜ ⊔ x = ⊤ := by rw [sup_comm, sup_compl_eq_top]
                                            /-
                                              🎉 no goals
                                            -/


theorem isCompl_compl : IsCompl x xᶜ :=
  IsCompl.of_eq inf_compl_eq_bot' sup_compl_eq_top


theorem sdiff_eq : x \ y = x ⊓ yᶜ :=
  BooleanAlgebra.sdiff_eq x y


theorem himp_eq : x ⇨ y = y ⊔ xᶜ :=
  BooleanAlgebra.himp_eq x y


instance (priority := 100) BooleanAlgebra.toComplementedLattice : ComplementedLattice α :=
  ⟨fun x => ⟨xᶜ, isCompl_compl⟩⟩

-- see Note [lower instance priority]

instance (priority := 100) BooleanAlgebra.toGeneralizedBooleanAlgebra :
    GeneralizedBooleanAlgebra α where
  __ := ‹BooleanAlgebra α›
                          /-
                            α : Type u
                            β : Type u_1
                            x y z : α
                            inst✝ : BooleanAlgebra α
                            a b : α
                            ⊢ Eq (Max.max (Min.min a b) (SDiff.sdiff a b)) a
                          -/
  sup_inf_sdiff a b := by rw [sdiff_eq, ← inf_sup_left, sup_compl_eq_top, inf_top_eq]
                          /-
                            🎉 no goals
                          -/
  inf_inf_sdiff a b := by
    /-
      α : Type u
      β : Type u_1
      x y z : α
      inst✝ : BooleanAlgebra α
      a b : α
      ⊢ Eq (Min.min (Min.min a b) (SDiff.sdiff a b)) Bot.bot
    -/
    rw [sdiff_eq, ← inf_inf_distrib_left, inf_compl_eq_bot', inf_bot_eq]
    /-
      🎉 no goals
    -/

-- See note [lower instance priority]

instance (priority := 100) BooleanAlgebra.toBiheytingAlgebra : BiheytingAlgebra α where
  __ := ‹BooleanAlgebra α›
  __ := GeneralizedBooleanAlgebra.toGeneralizedCoheytingAlgebra
  hnot := compl
                          /-
                            α : Type u
                            β : Type u_1
                            x y z : α
                            inst✝ : BooleanAlgebra α
                            a b c : α
                            ⊢ Iff (LE.le a (HImp.himp b c)) (LE.le (Min.min a b) c)
                          -/
  le_himp_iff a b c := by rw [himp_eq, isCompl_compl.le_sup_right_iff_inf_left_le]
                          /-
                            🎉 no goals
                          -/
  himp_bot _ := _root_.himp_eq.trans (bot_sup_eq _)
                    /-
                      α : Type u
                      β : Type u_1
                      x y z : α
                      inst✝ : BooleanAlgebra α
                      a : α
                      ⊢ Eq (SDiff.sdiff Top.top a) (HNot.hnot a)
                    -/
  top_sdiff a := by rw [sdiff_eq, top_inf_eq]; rfl
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem hnot_eq_compl : ￢x = xᶜ :=
  rfl

/- NOTE: Is this theorem needed at all or can we use `top_sdiff'`. -/

theorem top_sdiff : ⊤ \ x = xᶜ :=
  top_sdiff' x


theorem eq_compl_iff_isCompl : x = yᶜ ↔ IsCompl x y :=
  ⟨fun h => by
    /-
      α : Type u
      x y : α
      inst✝ : BooleanAlgebra α
      h : Eq x (HasCompl.compl y)
      ⊢ IsCompl x y
    -/
    rw [h]
    /-
      α : Type u
      x y : α
      inst✝ : BooleanAlgebra α
      h : Eq x (HasCompl.compl y)
      ⊢ IsCompl (HasCompl.compl y) y
    -/
    exact isCompl_compl.symm, IsCompl.eq_compl⟩
    /-
      🎉 no goals
    -/


theorem compl_eq_iff_isCompl : xᶜ = y ↔ IsCompl x y :=
  ⟨fun h => by
    /-
      α : Type u
      x y : α
      inst✝ : BooleanAlgebra α
      h : Eq (HasCompl.compl x) y
      ⊢ IsCompl x y
    -/
    rw [← h]
    /-
      α : Type u
      x y : α
      inst✝ : BooleanAlgebra α
      h : Eq (HasCompl.compl x) y
      ⊢ IsCompl x (HasCompl.compl x)
    -/
    exact isCompl_compl, IsCompl.compl_eq⟩
    /-
      🎉 no goals
    -/


theorem compl_eq_comm : xᶜ = y ↔ yᶜ = x := by
  /-
    α : Type u
    x y : α
    inst✝ : BooleanAlgebra α
    ⊢ Iff (Eq (HasCompl.compl x) y) (Eq (HasCompl.compl y) x)
  -/
  rw [eq_comm, compl_eq_iff_isCompl, eq_compl_iff_isCompl]
  /-
    🎉 no goals
  -/


theorem eq_compl_comm : x = yᶜ ↔ y = xᶜ := by
  /-
    α : Type u
    x y : α
    inst✝ : BooleanAlgebra α
    ⊢ Iff (Eq x (HasCompl.compl y)) (Eq y (HasCompl.compl x))
  -/
  rw [eq_comm, compl_eq_iff_isCompl, eq_compl_iff_isCompl]
  /-
    🎉 no goals
  -/


@[simp]
theorem compl_compl (x : α) : xᶜᶜ = x :=
  (@isCompl_compl _ x _).symm.compl_eq


theorem compl_comp_compl : compl ∘ compl = @id α :=
  funext compl_compl


@[simp]
theorem compl_involutive : Function.Involutive (compl : α → α) :=
  compl_compl


theorem compl_bijective : Function.Bijective (compl : α → α) :=
  compl_involutive.bijective


theorem compl_surjective : Function.Surjective (compl : α → α) :=
  compl_involutive.surjective


theorem compl_injective : Function.Injective (compl : α → α) :=
  compl_involutive.injective


@[simp]
theorem compl_inj_iff : xᶜ = yᶜ ↔ x = y :=
  compl_injective.eq_iff


theorem IsCompl.compl_eq_iff (h : IsCompl x y) : zᶜ = y ↔ z = x :=
  h.compl_eq ▸ compl_inj_iff


@[simp]
theorem compl_eq_top : xᶜ = ⊤ ↔ x = ⊥ :=
  isCompl_bot_top.compl_eq_iff


@[simp]
theorem compl_eq_bot : xᶜ = ⊥ ↔ x = ⊤ :=
  isCompl_top_bot.compl_eq_iff


@[simp]
theorem compl_inf : (x ⊓ y)ᶜ = xᶜ ⊔ yᶜ :=
  hnot_inf_distrib _ _


@[simp]
theorem compl_le_compl_iff_le : yᶜ ≤ xᶜ ↔ x ≤ y :=
               /-
                 α : Type u
                 x y : α
                 inst✝ : BooleanAlgebra α
                 h : LE.le (HasCompl.compl y) (HasCompl.compl x)
                 ⊢ LE.le x y
               -/
  ⟨fun h => by have h := compl_le_compl h; simpa using h, compl_le_compl⟩
                                           /-
                                             🎉 no goals
                                           -/


@[simp] lemma compl_lt_compl_iff_lt : yᶜ < xᶜ ↔ x < y :=
  lt_iff_lt_of_le_iff_le' compl_le_compl_iff_le compl_le_compl_iff_le


theorem compl_le_of_compl_le (h : yᶜ ≤ x) : xᶜ ≤ y := by
  /-
    α : Type u
    x y : α
    inst✝ : BooleanAlgebra α
    h : LE.le (HasCompl.compl y) x
    ⊢ LE.le (HasCompl.compl x) y
  -/
  simpa only [compl_compl] using compl_le_compl h
  /-
    🎉 no goals
  -/


theorem compl_le_iff_compl_le : xᶜ ≤ y ↔ yᶜ ≤ x :=
  ⟨compl_le_of_compl_le, compl_le_of_compl_le⟩


                                                     /-
                                                       α : Type u
                                                       x : α
                                                       inst✝ : BooleanAlgebra α
                                                       ⊢ Iff (LE.le (HasCompl.compl x) x) (Eq x Top.top)
                                                     -/
@[simp] theorem compl_le_self : xᶜ ≤ x ↔ x = ⊤ := by simpa using le_compl_self (a := xᶜ)
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp] theorem compl_lt_self [Nontrivial α] : xᶜ < x ↔ x = ⊤ := by
  /-
    α : Type u
    x : α
    inst✝¹ : BooleanAlgebra α
    inst✝ : Nontrivial α
    ⊢ Iff (LT.lt (HasCompl.compl x) x) (Eq x Top.top)
  -/
  simpa using lt_compl_self (a := xᶜ)
  /-
    🎉 no goals
  -/


@[simp]
                                           /-
                                             α : Type u
                                             x y : α
                                             inst✝ : BooleanAlgebra α
                                             ⊢ Eq (SDiff.sdiff x (HasCompl.compl y)) (Min.min x y)
                                           -/
theorem sdiff_compl : x \ yᶜ = x ⊓ y := by rw [sdiff_eq, compl_compl]
                                           /-
                                             🎉 no goals
                                           -/


instance OrderDual.instBooleanAlgebra : BooleanAlgebra αᵒᵈ where
  __ := instDistribLattice α
  __ := instHeytingAlgebra
  sdiff_eq _ _ := @himp_eq α _ _ _
  himp_eq _ _ := @sdiff_eq α _ _ _
  inf_compl_le_bot a := (@codisjoint_hnot_right _ _ (ofDual a)).top_le
  top_le_sup_compl a := (@disjoint_compl_right _ _ (ofDual a)).le_bot


@[simp]
                                                     /-
                                                       α : Type u
                                                       x y : α
                                                       inst✝ : BooleanAlgebra α
                                                       ⊢ Eq (Max.max (Min.min x y) (Min.min x (HasCompl.compl y))) x
                                                     -/
theorem sup_inf_inf_compl : x ⊓ y ⊔ x ⊓ yᶜ = x := by rw [← sdiff_eq, sup_inf_sdiff _ _]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem compl_sdiff : (x \ y)ᶜ = x ⇨ y := by
  /-
    α : Type u
    x y : α
    inst✝ : BooleanAlgebra α
    ⊢ Eq (HasCompl.compl (SDiff.sdiff x y)) (HImp.himp x y)
  -/
  rw [sdiff_eq, himp_eq, compl_inf, compl_compl, sup_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem compl_himp : (x ⇨ y)ᶜ = x \ y :=
  @compl_sdiff αᵒᵈ _ _ _


                                                  /-
                                                    α : Type u
                                                    x y : α
                                                    inst✝ : BooleanAlgebra α
                                                    ⊢ Eq (SDiff.sdiff (HasCompl.compl x) (HasCompl.compl y)) (SDiff.sdiff y x)
                                                  -/
theorem compl_sdiff_compl : xᶜ \ yᶜ = y \ x := by rw [sdiff_compl, sdiff_eq, inf_comm]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem compl_himp_compl : xᶜ ⇨ yᶜ = y ⇨ x :=
  @compl_sdiff_compl αᵒᵈ _ _ _


theorem disjoint_compl_left_iff : Disjoint xᶜ y ↔ y ≤ x := by
  /-
    α : Type u
    x y : α
    inst✝ : BooleanAlgebra α
    ⊢ Iff (Disjoint (HasCompl.compl x) y) (LE.le y x)
  -/
  rw [← le_compl_iff_disjoint_left, compl_compl]
  /-
    🎉 no goals
  -/


theorem disjoint_compl_right_iff : Disjoint x yᶜ ↔ x ≤ y := by
  /-
    α : Type u
    x y : α
    inst✝ : BooleanAlgebra α
    ⊢ Iff (Disjoint x (HasCompl.compl y)) (LE.le x y)
  -/
  rw [← le_compl_iff_disjoint_right, compl_compl]
  /-
    🎉 no goals
  -/


theorem codisjoint_himp_self_left : Codisjoint (x ⇨ y) x :=
  @disjoint_sdiff_self_left αᵒᵈ _ _ _


theorem codisjoint_himp_self_right : Codisjoint x (x ⇨ y) :=
  @disjoint_sdiff_self_right αᵒᵈ _ _ _


theorem himp_le : x ⇨ y ≤ z ↔ y ≤ z ∧ Codisjoint x z :=
  (@le_sdiff αᵒᵈ _ _ _ _).trans <| and_congr_right' <| @codisjoint_comm _ (_) _ _ _


@[simp] lemma himp_le_iff : x ⇨ y ≤ x ↔ x = ⊤ :=
  ⟨fun h ↦ codisjoint_self.1 <| codisjoint_himp_self_right.mono_right h, fun h ↦ le_top.trans h.ge⟩


@[simp] lemma himp_eq_left : x ⇨ y = x ↔ x = ⊤ ∧ y = ⊤ := by
  /-
    α : Type u
    x y : α
    inst✝ : BooleanAlgebra α
    ⊢ Iff (Eq (HImp.himp x y) x) (And (Eq x Top.top) (Eq y Top.top))
  -/
  rw [codisjoint_himp_self_left.eq_iff]; aesop
                                         /-
                                           🎉 no goals
                                         -/


lemma himp_ne_right : x ⇨ y ≠ x ↔ x ≠ ⊤ ∨ y ≠ ⊤ := himp_eq_left.not.trans not_and_or


instance Prop.instBooleanAlgebra : BooleanAlgebra Prop where
  __ := Prop.instHeytingAlgebra
  __ := GeneralizedHeytingAlgebra.toDistribLattice
  compl := Not
  himp_eq _ _ := propext imp_iff_or_not
  inf_compl_le_bot _ H := H.2 H.1
  top_le_sup_compl p _ := Classical.em p


instance Prod.instBooleanAlgebra [BooleanAlgebra α] [BooleanAlgebra β] :
    BooleanAlgebra (α × β) where
  __ := instDistribLattice α β
  __ := instHeytingAlgebra
                    /-
                      α : Type u
                      β : Type u_1
                      x✝ y✝ z : α
                      inst✝¹ : BooleanAlgebra α
                      inst✝ : BooleanAlgebra β
                      x y : Prod α β
                      ⊢ Eq (HImp.himp x y) (Max.max y (HasCompl.compl x))
                    -/
                     /-
                       α : Type u
                       β : Type u_1
                       x✝ y✝ z : α
                       inst✝¹ : BooleanAlgebra α
                       inst✝ : BooleanAlgebra β
                       x y : Prod α β
                       ⊢ Eq (SDiff.sdiff x y) (Min.min x (HasCompl.compl y))
                     -/
                           /-
                             α : Type u
                             β : Type u_1
                             x✝ y z : α
                             inst✝¹ : BooleanAlgebra α
                             inst✝ : BooleanAlgebra β
                             x : Prod α β
                             ⊢ LE.le (Min.min x (HasCompl.compl x)) Bot.bot
                           -/
                                           /-
                                             🎉 no goals
                                           -/
                             /-
                               🎉 no goals
                             -/
                                           /-
                                             🎉 no goals
                                           -/
                           /-
                             α : Type u
                             β : Type u_1
                             x✝ y z : α
                             inst✝¹ : BooleanAlgebra α
                             inst✝ : BooleanAlgebra β
                             x : Prod α β
                             ⊢ LE.le Top.top (Max.max x (HasCompl.compl x))
                           -/
                                           /-
                                             🎉 no goals
                                           -/
                            /-
                              🎉 no goals
                            -/
                                           /-
                                             🎉 no goals
                                           -/
                             /-
                               🎉 no goals
                             -/
  himp_eq x y := by ext <;> simp [himp_eq]
                            /-
                              🎉 no goals
                            -/
  sdiff_eq x y := by ext <;> simp [sdiff_eq]
  inf_compl_le_bot x := by constructor <;> simp
  top_le_sup_compl x := by constructor <;> simp


instance Pi.instBooleanAlgebra {ι : Type u} {α : ι → Type v} [∀ i, BooleanAlgebra (α i)] :
    BooleanAlgebra (∀ i, α i) where
  __ := instDistribLattice
  __ := instHeytingAlgebra
  sdiff_eq _ _ := funext fun _ => sdiff_eq
  himp_eq _ _ := funext fun _ => himp_eq
  inf_compl_le_bot _ _ := BooleanAlgebra.inf_compl_le_bot _
  top_le_sup_compl _ _ := BooleanAlgebra.top_le_sup_compl _


instance Bool.instBooleanAlgebra : BooleanAlgebra Bool where
  __ := instDistribLattice
  __ := linearOrder
  __ := instBoundedOrder
  compl := not
  inf_compl_le_bot a := a.and_not_self.le
  top_le_sup_compl a := a.or_not_self.ge


                                             /-
                                               ⊢ Eq (fun x1 x2 => Max.max x1 x2) Bool.or
                                             -/
theorem Bool.sup_eq_bor : (· ⊔ ·) = or := by dsimp
                                             /-
                                               🎉 no goals
                                             -/


                                               /-
                                                 ⊢ Eq (fun x1 x2 => Min.min x1 x2) Bool.and
                                               -/
theorem Bool.inf_eq_band : (· ⊓ ·) = and := by dsimp
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem Bool.compl_eq_bnot : HasCompl.compl = not :=
  rfl


/-- Pullback a `GeneralizedBooleanAlgebra` along an injection. -/
protected abbrev Function.Injective.generalizedBooleanAlgebra [Max α] [Min α] [Bot α] [SDiff α]
    [GeneralizedBooleanAlgebra β] (f : α → β) (hf : Injective f)
    (map_sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b) (map_inf : ∀ a b, f (a ⊓ b) = f a ⊓ f b)
    (map_bot : f ⊥ = ⊥) (map_sdiff : ∀ a b, f (a \ b) = f a \ f b) :
    GeneralizedBooleanAlgebra α where
  __ := hf.generalizedCoheytingAlgebra f map_sup map_inf map_bot map_sdiff
  __ := hf.distribLattice f map_sup map_inf
                                /-
                                  α : Type u
                                  β : Type u_1
                                  x y z : α
                                  inst✝⁴ : Max α
                                  inst✝³ : Min α
                                  inst✝² : Bot α
                                  inst✝¹ : SDiff α
                                  inst✝ : GeneralizedBooleanAlgebra β
                                  f : α → β
                                  hf : Function.Injective f
                                  map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
                                  map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
                                  map_bot : Eq (f Bot.bot) Bot.bot
                                  map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
                                  a b : α
                                  ⊢ Eq (f (Max.max (Min.min a b) (SDiff.sdiff a b))) (f a)
                                -/
  sup_inf_sdiff a b := hf <| by rw [map_sup, map_sdiff, map_inf, sup_inf_sdiff]
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  α : Type u
                                  β : Type u_1
                                  x y z : α
                                  inst✝⁴ : Max α
                                  inst✝³ : Min α
                                  inst✝² : Bot α
                                  inst✝¹ : SDiff α
                                  inst✝ : GeneralizedBooleanAlgebra β
                                  f : α → β
                                  hf : Function.Injective f
                                  map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
                                  map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
                                  map_bot : Eq (f Bot.bot) Bot.bot
                                  map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
                                  a b : α
                                  ⊢ Eq (f (Min.min (Min.min a b) (SDiff.sdiff a b))) (f Bot.bot)
                                -/
  inf_inf_sdiff a b := hf <| by rw [map_inf, map_sdiff, map_inf, inf_inf_sdiff, map_bot]
                                /-
                                  🎉 no goals
                                -/

-- See note [reducible non-instances]

/-- Pullback a `BooleanAlgebra` along an injection. -/
protected abbrev Function.Injective.booleanAlgebra [Max α] [Min α] [Top α] [Bot α] [HasCompl α]
    [SDiff α] [HImp α] [BooleanAlgebra β] (f : α → β) (hf : Injective f)
    (map_sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b) (map_inf : ∀ a b, f (a ⊓ b) = f a ⊓ f b)
    (map_top : f ⊤ = ⊤) (map_bot : f ⊥ = ⊥) (map_compl : ∀ a, f aᶜ = (f a)ᶜ)
    (map_sdiff : ∀ a b, f (a \ b) = f a \ f b) (map_himp : ∀ a b, f (a ⇨ b) = f a ⇨ f b) :
    BooleanAlgebra α where
  __ := hf.generalizedBooleanAlgebra f map_sup map_inf map_bot map_sdiff
  compl := compl
  himp := himp
  top := ⊤
  le_top _ := (@le_top β _ _ _).trans map_top.ge
  bot_le _ := map_bot.le.trans bot_le
                                                   /-
                                                     α : Type u
                                                     β : Type u_1
                                                     x y z : α
                                                     inst✝⁷ : Max α
                                                     inst✝⁶ : Min α
                                                     inst✝⁵ : Top α
                                                     inst✝⁴ : Bot α
                                                     inst✝³ : HasCompl α
                                                     inst✝² : SDiff α
                                                     inst✝¹ : HImp α
                                                     inst✝ : BooleanAlgebra β
                                                     f : α → β
                                                     hf : Function.Injective f
                                                     map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
                                                     map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
                                                     map_top : Eq (f Top.top) Top.top
                                                     map_bot : Eq (f Bot.bot) Bot.bot
                                                     map_compl : ∀ (a : α), Eq (f (HasCompl.compl a)) (HasCompl.compl (f a))
                                                     map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
                                                     map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
                                                     a : α
                                                     ⊢ Eq (Min.min (f a) (f (HasCompl.compl a))) (f Bot.bot)
                                                   -/
  inf_compl_le_bot a := ((map_inf _ _).trans <| by rw [map_compl, inf_compl_eq_bot, map_bot]).le
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                   /-
                                                     α : Type u
                                                     β : Type u_1
                                                     x y z : α
                                                     inst✝⁷ : Max α
                                                     inst✝⁶ : Min α
                                                     inst✝⁵ : Top α
                                                     inst✝⁴ : Bot α
                                                     inst✝³ : HasCompl α
                                                     inst✝² : SDiff α
                                                     inst✝¹ : HImp α
                                                     inst✝ : BooleanAlgebra β
                                                     f : α → β
                                                     hf : Function.Injective f
                                                     map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
                                                     map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
                                                     map_top : Eq (f Top.top) Top.top
                                                     map_bot : Eq (f Bot.bot) Bot.bot
                                                     map_compl : ∀ (a : α), Eq (f (HasCompl.compl a)) (HasCompl.compl (f a))
                                                     map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
                                                     map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
                                                     a : α
                                                     ⊢ Eq (Max.max (f a) (f (HasCompl.compl a))) (f Top.top)
                                                   -/
  top_le_sup_compl a := ((map_sup _ _).trans <| by rw [map_compl, sup_compl_eq_top, map_top]).ge
                                                   /-
                                                     🎉 no goals
                                                   -/
  sdiff_eq a b := by
    /-
      α : Type u
      β : Type u_1
      x y z : α
      inst✝⁷ : Max α
      inst✝⁶ : Min α
      inst✝⁵ : Top α
      inst✝⁴ : Bot α
      inst✝³ : HasCompl α
      inst✝² : SDiff α
      inst✝¹ : HImp α
      inst✝ : BooleanAlgebra β
      f : α → β
      hf : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      map_top : Eq (f Top.top) Top.top
      map_bot : Eq (f Bot.bot) Bot.bot
      map_compl : ∀ (a : α), Eq (f (HasCompl.compl a)) (HasCompl.compl (f a))
      map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
      map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
      a b : α
      ⊢ Eq (SDiff.sdiff a b) (Min.min a (HasCompl.compl b))
    -/
    refine hf ((map_sdiff _ _).trans (sdiff_eq.trans ?_))
    /-
      α : Type u
      β : Type u_1
      x y z : α
      inst✝⁷ : Max α
      inst✝⁶ : Min α
      inst✝⁵ : Top α
      inst✝⁴ : Bot α
      inst✝³ : HasCompl α
      inst✝² : SDiff α
      inst✝¹ : HImp α
      inst✝ : BooleanAlgebra β
      f : α → β
      hf : Function.Injective f
      map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
      map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
      map_top : Eq (f Top.top) Top.top
      map_bot : Eq (f Bot.bot) Bot.bot
      map_compl : ∀ (a : α), Eq (f (HasCompl.compl a)) (HasCompl.compl (f a))
      map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
      map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
      a b : α
      ⊢ Eq (Min.min (f a) (HasCompl.compl (f b))) (f (Min.min a (HasCompl.compl b)))
    -/
    rw [map_inf, map_compl]
    /-
      🎉 no goals
    -/
                                                                   /-
                                                                     α : Type u
                                                                     β : Type u_1
                                                                     x y z : α
                                                                     inst✝⁷ : Max α
                                                                     inst✝⁶ : Min α
                                                                     inst✝⁵ : Top α
                                                                     inst✝⁴ : Bot α
                                                                     inst✝³ : HasCompl α
                                                                     inst✝² : SDiff α
                                                                     inst✝¹ : HImp α
                                                                     inst✝ : BooleanAlgebra β
                                                                     f : α → β
                                                                     hf : Function.Injective f
                                                                     map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
                                                                     map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
                                                                     map_top : Eq (f Top.top) Top.top
                                                                     map_bot : Eq (f Bot.bot) Bot.bot
                                                                     map_compl : ∀ (a : α), Eq (f (HasCompl.compl a)) (HasCompl.compl (f a))
                                                                     map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
                                                                     map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
                                                                     a b : α
                                                                     ⊢ Eq (Max.max (f b) (HasCompl.compl (f a))) (f (Max.max b (HasCompl.compl a)))
                                                                   -/
  himp_eq a b := hf <| (map_himp _ _).trans <| himp_eq.trans <| by rw [map_sup, map_compl]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


instance PUnit.instBooleanAlgebra : BooleanAlgebra PUnit where
  __ := PUnit.instBiheytingAlgebra
                   /-
                     α : Type u
                     β : Type u_1
                     x y z : α
                     ⊢ ∀ (x y z : PUnit.{?u.92137 + 1}), LE.le (Min.min (Max.max x y) (Max.max x z) …
                   -/
  le_sup_inf := by simp
                   /-
                     🎉 no goals
                   -/
  inf_compl_le_bot _ := trivial
  top_le_sup_compl _ := trivial


/--
An alternative constructor for boolean algebras:
a distributive lattice that is complemented is a boolean algebra.

This is not an instance, because it creates data using choice.
-/
noncomputable
def booleanAlgebraOfComplemented [BoundedOrder α] [ComplementedLattice α] : BooleanAlgebra α where
  __ := (inferInstanceAs (DistribLattice α))
  __ := (inferInstanceAs (BoundedOrder α))
  compl a := Classical.choose <| exists_isCompl a
  inf_compl_le_bot a := (Classical.choose_spec (exists_isCompl a)).disjoint.le_bot
  top_le_sup_compl a := (Classical.choose_spec (exists_isCompl a)).codisjoint.top_le



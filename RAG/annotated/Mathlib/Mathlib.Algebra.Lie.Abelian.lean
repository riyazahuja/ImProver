/-- A Lie (ring) module is trivial iff all brackets vanish. -/
class LieModule.IsTrivial (L : Type v) (M : Type w) [Bracket L M] [Zero M] : Prop where
  trivial : ∀ (x : L) (m : M), ⁅x, m⁆ = 0


@[simp]
theorem trivial_lie_zero (L : Type v) (M : Type w) [Bracket L M] [Zero M] [LieModule.IsTrivial L M]
    (x : L) (m : M) : ⁅x, m⁆ = 0 :=
  LieModule.IsTrivial.trivial x m


instance LieModule.instIsTrivialOfSubsingleton {L M : Type*}
    [LieRing L] [AddCommGroup M] [LieRingModule L M] [Subsingleton L] : LieModule.IsTrivial L M :=
                /-
                  L : Type u_1
                  M : Type u_2
                  inst✝³ : LieRing L
                  inst✝² : AddCommGroup M
                  inst✝¹ : LieRingModule L M
                  inst✝ : Subsingleton L
                  x : L
                  m : M
                  ⊢ Eq (Bracket.bracket x m) 0
                -/
  ⟨fun x m ↦ by rw [Subsingleton.eq_zero x, zero_lie]⟩
                /-
                  🎉 no goals
                -/


instance LieModule.instIsTrivialOfSubsingleton' {L M : Type*}
    [LieRing L] [AddCommGroup M] [LieRingModule L M] [Subsingleton M] : LieModule.IsTrivial L M :=
                /-
                  L : Type u_1
                  M : Type u_2
                  inst✝³ : LieRing L
                  inst✝² : AddCommGroup M
                  inst✝¹ : LieRingModule L M
                  inst✝ : Subsingleton M
                  x : L
                  m : M
                  ⊢ Eq (Bracket.bracket x m) 0
                -/
  ⟨fun x m ↦ by simp_rw [Subsingleton.eq_zero m, lie_zero]⟩
                /-
                  🎉 no goals
                -/


/-- A Lie algebra is Abelian iff it is trivial as a Lie module over itself. -/
abbrev IsLieAbelian (L : Type v) [Bracket L L] [Zero L] : Prop :=
  LieModule.IsTrivial L L


instance LieIdeal.isLieAbelian_of_trivial (R : Type u) (L : Type v) [CommRing R] [LieRing L]
    [LieAlgebra R L] (I : LieIdeal R L) [h : LieModule.IsTrivial L I] : IsLieAbelian I where
                    /-
                      R : Type u
                      L : Type v
                      inst✝² : CommRing R
                      inst✝¹ : LieRing L
                      inst✝ : LieAlgebra R L
                      I : LieIdeal R L
                      h : LieModule.IsTrivial L (Subtype fun x => Membership.mem I x)
                      x y : Subtype fun x => Membership.mem I x
                      ⊢ Eq (Bracket.bracket x y) 0
                    -/
  trivial x y := by apply h.trivial
                    /-
                      🎉 no goals
                    -/


theorem Function.Injective.isLieAbelian {R : Type u} {L₁ : Type v} {L₂ : Type w} [CommRing R]
    [LieRing L₁] [LieRing L₂] [LieAlgebra R L₁] [LieAlgebra R L₂] {f : L₁ →ₗ⁅R⁆ L₂}
    (h₁ : Function.Injective f) (_ : IsLieAbelian L₂) : IsLieAbelian L₁ :=
  { trivial := fun x y => h₁ <|
      calc
        f ⁅x, y⁆ = ⁅f x, f y⁆ := LieHom.map_lie f x y
        _ = 0 := trivial_lie_zero _ _ _ _
        _ = f 0 := f.map_zero.symm}


theorem Function.Surjective.isLieAbelian {R : Type u} {L₁ : Type v} {L₂ : Type w} [CommRing R]
    [LieRing L₁] [LieRing L₂] [LieAlgebra R L₁] [LieAlgebra R L₂] {f : L₁ →ₗ⁅R⁆ L₂}
    (h₁ : Function.Surjective f) (h₂ : IsLieAbelian L₁) : IsLieAbelian L₂ :=
  { trivial := fun x y => by
      /-
        R : Type u
        L₁ : Type v
        L₂ : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L₁
        inst✝² : LieRing L₂
        inst✝¹ : LieAlgebra R L₁
        inst✝ : LieAlgebra R L₂
        f : LieHom R L₁ L₂
        h₁ : Function.Surjective ⇑f
        h₂ : IsLieAbelian L₁
        x y : L₂
        ⊢ Eq (Bracket.bracket x y) 0
      -/
      obtain ⟨u, rfl⟩ := h₁ x
      /-
        case intro
        R : Type u
        L₁ : Type v
        L₂ : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L₁
        inst✝² : LieRing L₂
        inst✝¹ : LieAlgebra R L₁
        inst✝ : LieAlgebra R L₂
        f : LieHom R L₁ L₂
        h₁ : Function.Surjective ⇑f
        h₂ : IsLieAbelian L₁
        y : L₂
        u : L₁
        ⊢ Eq (Bracket.bracket (f u) y) 0
      -/
      obtain ⟨v, rfl⟩ := h₁ y
      /-
        case intro.intro
        R : Type u
        L₁ : Type v
        L₂ : Type w
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L₁
        inst✝² : LieRing L₂
        inst✝¹ : LieAlgebra R L₁
        inst✝ : LieAlgebra R L₂
        f : LieHom R L₁ L₂
        h₁ : Function.Surjective ⇑f
        h₂ : IsLieAbelian L₁
        u v : L₁
        ⊢ Eq (Bracket.bracket (f u) (f v)) 0
      -/
      rw [← LieHom.map_lie, trivial_lie_zero, LieHom.map_zero] }
      /-
        🎉 no goals
      -/


theorem lie_abelian_iff_equiv_lie_abelian {R : Type u} {L₁ : Type v} {L₂ : Type w} [CommRing R]
    [LieRing L₁] [LieRing L₂] [LieAlgebra R L₁] [LieAlgebra R L₂] (e : L₁ ≃ₗ⁅R⁆ L₂) :
    IsLieAbelian L₁ ↔ IsLieAbelian L₂ :=
  ⟨e.symm.injective.isLieAbelian, e.injective.isLieAbelian⟩


theorem commutative_ring_iff_abelian_lie_ring {A : Type v} [Ring A] :
    Std.Commutative (α := A) (· * ·) ↔ IsLieAbelian A := by
  have h₁ : Std.Commutative (α := A) (· * ·) ↔ ∀ a b : A, a * b = b * a :=
    ⟨fun h => h.1, fun h => ⟨h⟩⟩
  /-
    A : Type v
    inst✝ : Ring A
    h₁ : Iff (Std.Commutative fun x1 x2 => HMul.hMul x1 x2) (∀ (a b : A), Eq (HMul …
    ⊢ Iff (Std.Commutative fun x1 x2 => HMul.hMul x1 x2) (IsLieAbelian A)
  -/
  have h₂ : IsLieAbelian A ↔ ∀ a b : A, ⁅a, b⁆ = 0 := ⟨fun h => h.1, fun h => ⟨h⟩⟩
  /-
    A : Type v
    inst✝ : Ring A
    h₁ : Iff (Std.Commutative fun x1 x2 => HMul.hMul x1 x2) (∀ (a b : A), Eq (HMul …
    h₂ : Iff (IsLieAbelian A) (∀ (a b : A), Eq (Bracket.bracket a b) 0)
    ⊢ Iff (Std.Commutative fun x1 x2 => HMul.hMul x1 x2) (IsLieAbelian A)
  -/
  simp only [h₁, h₂, LieRing.of_associative_ring_bracket, sub_eq_zero]
  /-
    🎉 no goals
  -/


/-- The kernel of the action of a Lie algebra `L` on a Lie module `M` as a Lie ideal in `L`. -/
protected def ker : LieIdeal R L :=
  (toEnd R L M).ker


@[simp]
protected theorem mem_ker (x : L) : x ∈ LieModule.ker R L M ↔ ∀ m : M, ⁅x, m⁆ = 0 := by
  simp only [LieModule.ker, LieHom.mem_ker, LinearMap.ext_iff, LinearMap.zero_apply,
    toEnd_apply_apply]


/-- The largest submodule of a Lie module `M` on which the Lie algebra `L` acts trivially. -/
def maxTrivSubmodule : LieSubmodule R L M where
  carrier := { m | ∀ x : L, ⁅x, m⁆ = 0 }
  zero_mem' x := lie_zero x
                               /-
                                 R : Type u
                                 L : Type v
                                 M : Type w
                                 N : Type w₁
                                 inst✝¹⁰ : CommRing R
                                 inst✝⁹ : LieRing L
                                 inst✝⁸ : LieAlgebra R L
                                 inst✝⁷ : AddCommGroup M
                                 inst✝⁶ : Module R M
                                 inst✝⁵ : LieRingModule L M
                                 inst✝⁴ : LieModule R L M
                                 inst✝³ : AddCommGroup N
                                 inst✝² : Module R N
                                 inst✝¹ : LieRingModule L N
                                 inst✝ : LieModule R L N
                                 x y : M
                                 hx : Membership.mem (setOf fun m => ∀ (x : L), Eq (Bracket.bracket x m) 0) x
                                 hy : Membership.mem (setOf fun m => ∀ (x : L), Eq (Bracket.bracket x m) 0) y
                                 z : L
                                 ⊢ Eq (Bracket.bracket z (HAdd.hAdd x y)) 0
                               -/
  add_mem' {x y} hx hy z := by rw [lie_add, hx, hy, add_zero]
                               /-
                                 🎉 no goals
                               -/
                           /-
                             R : Type u
                             L : Type v
                             M : Type w
                             N : Type w₁
                             inst✝¹⁰ : CommRing R
                             inst✝⁹ : LieRing L
                             inst✝⁸ : LieAlgebra R L
                             inst✝⁷ : AddCommGroup M
                             inst✝⁶ : Module R M
                             inst✝⁵ : LieRingModule L M
                             inst✝⁴ : LieModule R L M
                             inst✝³ : AddCommGroup N
                             inst✝² : Module R N
                             inst✝¹ : LieRingModule L N
                             inst✝ : LieModule R L N
                             c : R
                             x : M
                             hx : Membership.mem { carrier := setOf fun m => ∀ (x : L), Eq (Bracket.bracket …
                             y : L
                             ⊢ Eq (Bracket.bracket y (HSMul.hSMul c x)) 0
                           -/
  smul_mem' c x hx y := by rw [lie_smul, hx, smul_zero]
                           /-
                             🎉 no goals
                           -/
                           /-
                             R : Type u
                             L : Type v
                             M : Type w
                             N : Type w₁
                             inst✝¹⁰ : CommRing R
                             inst✝⁹ : LieRing L
                             inst✝⁸ : LieAlgebra R L
                             inst✝⁷ : AddCommGroup M
                             inst✝⁶ : Module R M
                             inst✝⁵ : LieRingModule L M
                             inst✝⁴ : LieModule R L M
                             inst✝³ : AddCommGroup N
                             inst✝² : Module R N
                             inst✝¹ : LieRingModule L N
                             inst✝ : LieModule R L N
                             x : L
                             m : M
                             hm : Membership.mem { carrier := setOf fun m => ∀ (x : L), Eq (Bracket.bracket …
                             y : L
                             ⊢ Eq (Bracket.bracket y (Bracket.bracket x m)) 0
                           -/
  lie_mem {x m} hm y := by rw [hm, lie_zero]
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem mem_maxTrivSubmodule (m : M) : m ∈ maxTrivSubmodule R L M ↔ ∀ x : L, ⁅x, m⁆ = 0 :=
  Iff.rfl


instance : IsTrivial L (maxTrivSubmodule R L M) where trivial x m := Subtype.ext (m.property x)


@[simp]
theorem ideal_oper_maxTrivSubmodule_eq_bot (I : LieIdeal R L) :
    ⁅I, maxTrivSubmodule R L M⁆ = ⊥ := by
  rw [← LieSubmodule.toSubmodule_inj, LieSubmodule.lieIdeal_oper_eq_linear_span,
    LieSubmodule.bot_toSubmodule, Submodule.span_eq_bot]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    I : LieIdeal R L
    ⊢ ∀ (x : M), Membership.mem (setOf fun m => Exists fun x => Exists fun n => Eq …
  -/
  rintro m ⟨⟨x, hx⟩, ⟨⟨m, hm⟩, rfl⟩⟩
  /-
    case intro.mk.intro.mk
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    I : LieIdeal R L
    x : L
    hx : Membership.mem I x
    m : M
    hm : Membership.mem (LieModule.maxTrivSubmodule R L M) m
    ⊢ Eq (Bracket.bracket ↑⟨x, hx⟩ ↑⟨m, hm⟩) 0
  -/
  exact hm x
  /-
    🎉 no goals
  -/


theorem le_max_triv_iff_bracket_eq_bot {N : LieSubmodule R L M} :
    N ≤ maxTrivSubmodule R L M ↔ ⁅(⊤ : LieIdeal R L), N⁆ = ⊥ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    N : LieSubmodule R L M
    ⊢ Iff (LE.le N (LieModule.maxTrivSubmodule R L M)) (Eq (Bracket.bracket Top.to …
  -/
  refine ⟨fun h => ?_, fun h m hm => ?_⟩
    /-
      case refine_1
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      N : LieSubmodule R L M
      h : LE.le N (LieModule.maxTrivSubmodule R L M)
      ⊢ Eq (Bracket.bracket Top.top N) Bot.bot
    -/
  · rw [← le_bot_iff, ← ideal_oper_maxTrivSubmodule_eq_bot R L M ⊤]
    /-
      case refine_1
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      N : LieSubmodule R L M
      h : LE.le N (LieModule.maxTrivSubmodule R L M)
      ⊢ LE.le (Bracket.bracket Top.top N) (Bracket.bracket Top.top (LieModule.maxTri …
    -/
    exact LieSubmodule.mono_lie_right ⊤ h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      N : LieSubmodule R L M
      h : Eq (Bracket.bracket Top.top N) Bot.bot
      m : M
      hm : Membership.mem N m
      ⊢ Membership.mem (LieModule.maxTrivSubmodule R L M) m
    -/
  · rw [mem_maxTrivSubmodule]
    /-
      case refine_2
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      N : LieSubmodule R L M
      h : Eq (Bracket.bracket Top.top N) Bot.bot
      m : M
      hm : Membership.mem N m
      ⊢ ∀ (x : L), Eq (Bracket.bracket x m) 0
    -/
    rw [LieSubmodule.lie_eq_bot_iff] at h
    /-
      case refine_2
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      N : LieSubmodule R L M
      h : ∀ (x : L), Membership.mem Top.top x → ∀ (m : M), Membership.mem N m → Eq ( …
      m : M
      hm : Membership.mem N m
      ⊢ ∀ (x : L), Eq (Bracket.bracket x m) 0
    -/
    exact fun x => h x (LieSubmodule.mem_top x) m hm
    /-
      🎉 no goals
    -/


theorem trivial_iff_le_maximal_trivial (N : LieSubmodule R L M) :
    IsTrivial L N ↔ N ≤ maxTrivSubmodule R L M :=
  ⟨fun h m hm x => IsTrivial.casesOn h fun h => Subtype.ext_iff.mp (h x ⟨m, hm⟩), fun h =>
    { trivial := fun x m => Subtype.ext (h m.2 x) }⟩


theorem isTrivial_iff_max_triv_eq_top : IsTrivial L M ↔ maxTrivSubmodule R L M = ⊤ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    ⊢ Iff (LieModule.IsTrivial L M) (Eq (LieModule.maxTrivSubmodule R L M) Top.top)
  -/
  constructor
    /-
      case mp
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      ⊢ LieModule.IsTrivial L M → Eq (LieModule.maxTrivSubmodule R L M) Top.top
    -/
  · rintro ⟨h⟩; ext; simp only [mem_maxTrivSubmodule, h, forall_const, LieSubmodule.mem_top]
                     /-
                       🎉 no goals
                     -/
    /-
      case mpr
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      ⊢ Eq (LieModule.maxTrivSubmodule R L M) Top.top → LieModule.IsTrivial L M
    -/
  · intro h; constructor; intro x m; revert x
    /-
      case mpr.trivial
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      h : Eq (LieModule.maxTrivSubmodule R L M) Top.top
      m : M
      ⊢ ∀ (x : L), Eq (Bracket.bracket x m) 0
    -/
    rw [← mem_maxTrivSubmodule R L M, h]; exact LieSubmodule.mem_top m
                                          /-
                                            🎉 no goals
                                          -/


/-- `maxTrivSubmodule` is functorial. -/
def maxTrivHom (f : M →ₗ⁅R,L⁆ N) : maxTrivSubmodule R L M →ₗ⁅R,L⁆ maxTrivSubmodule R L N where
  toFun m := ⟨f m, fun x =>
    (LieModuleHom.map_lie _ _ _).symm.trans <|
      (congr_arg f (m.property x)).trans (LieModuleHom.map_zero _)⟩
                     /-
                       R : Type u
                       L : Type v
                       M : Type w
                       N : Type w₁
                       inst✝¹⁰ : CommRing R
                       inst✝⁹ : LieRing L
                       inst✝⁸ : LieAlgebra R L
                       inst✝⁷ : AddCommGroup M
                       inst✝⁶ : Module R M
                       inst✝⁵ : LieRingModule L M
                       inst✝⁴ : LieModule R L M
                       inst✝³ : AddCommGroup N
                       inst✝² : Module R N
                       inst✝¹ : LieRingModule L N
                       inst✝ : LieModule R L N
                       f : LieModuleHom R L M N
                       m n : Subtype fun x => Membership.mem (LieModule.maxTrivSubmodule R L M) x
                       ⊢ Eq ((fun m => ⟨f ↑m, ⋯⟩) (HAdd.hAdd m n)) (HAdd.hAdd ((fun m => ⟨f ↑m, ⋯⟩) m …
                     -/
  map_add' m n := by simp [Function.comp_apply]; rfl -- Porting note:
                                                 /-
                                                   🎉 no goals
                                                 -/
                      /-
                        R : Type u
                        L : Type v
                        M : Type w
                        N : Type w₁
                        inst✝¹⁰ : CommRing R
                        inst✝⁹ : LieRing L
                        inst✝⁸ : LieAlgebra R L
                        inst✝⁷ : AddCommGroup M
                        inst✝⁶ : Module R M
                        inst✝⁵ : LieRingModule L M
                        inst✝⁴ : LieModule R L M
                        inst✝³ : AddCommGroup N
                        inst✝² : Module R N
                        inst✝¹ : LieRingModule L N
                        inst✝ : LieModule R L N
                        f : LieModuleHom R L M N
                        t : R
                        m : Subtype fun x => Membership.mem (LieModule.maxTrivSubmodule R L M) x
                        ⊢ Eq ({ toFun := fun m => ⟨f ↑m, ⋯⟩, map_add' := ⋯ }.toFun (HSMul.hSMul t m))  …
                      -/
  map_smul' t m := by simp [Function.comp_apply]; rfl -- these two were `by simpa`
                                                  /-
                                                    🎉 no goals
                                                  -/
                       /-
                         R : Type u
                         L : Type v
                         M : Type w
                         N : Type w₁
                         inst✝¹⁰ : CommRing R
                         inst✝⁹ : LieRing L
                         inst✝⁸ : LieAlgebra R L
                         inst✝⁷ : AddCommGroup M
                         inst✝⁶ : Module R M
                         inst✝⁵ : LieRingModule L M
                         inst✝⁴ : LieModule R L M
                         inst✝³ : AddCommGroup N
                         inst✝² : Module R N
                         inst✝¹ : LieRingModule L N
                         inst✝ : LieModule R L N
                         f : LieModuleHom R L M N
                         x : L
                         m : Subtype fun x => Membership.mem (LieModule.maxTrivSubmodule R L M) x
                         ⊢ Eq ({ toFun := fun m => ⟨f ↑m, ⋯⟩, map_add' := ⋯, map_smul' := ⋯ }.toFun (Br …
                       -/
  map_lie' {x m} := by simp
                       /-
                         🎉 no goals
                       -/


@[norm_cast, simp]
theorem coe_maxTrivHom_apply (f : M →ₗ⁅R,L⁆ N) (m : maxTrivSubmodule R L M) :
    (maxTrivHom f m : N) = f m :=
  rfl


/-- The maximal trivial submodules of Lie-equivalent Lie modules are Lie-equivalent. -/
def maxTrivEquiv (e : M ≃ₗ⁅R,L⁆ N) : maxTrivSubmodule R L M ≃ₗ⁅R,L⁆ maxTrivSubmodule R L N :=
  { maxTrivHom (e : M →ₗ⁅R,L⁆ N) with
    toFun := maxTrivHom (e : M →ₗ⁅R,L⁆ N)
    invFun := maxTrivHom (e.symm : N →ₗ⁅R,L⁆ M)
                            /-
                              R : Type u
                              L : Type v
                              M : Type w
                              N : Type w₁
                              inst✝¹⁰ : CommRing R
                              inst✝⁹ : LieRing L
                              inst✝⁸ : LieAlgebra R L
                              inst✝⁷ : AddCommGroup M
                              inst✝⁶ : Module R M
                              inst✝⁵ : LieRingModule L M
                              inst✝⁴ : LieModule R L M
                              inst✝³ : AddCommGroup N
                              inst✝² : Module R N
                              inst✝¹ : LieRingModule L N
                              inst✝ : LieModule R L N
                              e : LieModuleEquiv R L M N
                              m : Subtype fun x => Membership.mem (LieModule.maxTrivSubmodule R L M) x
                              ⊢ Eq ((LieModule.maxTrivHom e.symm.toLieModuleHom) ((↑{ toFun := ⇑(LieModule.m …
                            -/
    left_inv := fun m => by ext; simp [LieModuleEquiv.coe_toLieModuleHom]
                                 /-
                                   🎉 no goals
                                 -/
                             /-
                               R : Type u
                               L : Type v
                               M : Type w
                               N : Type w₁
                               inst✝¹⁰ : CommRing R
                               inst✝⁹ : LieRing L
                               inst✝⁸ : LieAlgebra R L
                               inst✝⁷ : AddCommGroup M
                               inst✝⁶ : Module R M
                               inst✝⁵ : LieRingModule L M
                               inst✝⁴ : LieModule R L M
                               inst✝³ : AddCommGroup N
                               inst✝² : Module R N
                               inst✝¹ : LieRingModule L N
                               inst✝ : LieModule R L N
                               e : LieModuleEquiv R L M N
                               n : Subtype fun x => Membership.mem (LieModule.maxTrivSubmodule R L N) x
                               ⊢ Eq ((↑{ toFun := ⇑(LieModule.maxTrivHom e.toLieModuleHom), map_add' := ⋯, ma …
                             -/
    right_inv := fun n => by ext; simp [LieModuleEquiv.coe_toLieModuleHom] }
                                  /-
                                    🎉 no goals
                                  -/


@[norm_cast, simp]
theorem coe_maxTrivEquiv_apply (e : M ≃ₗ⁅R,L⁆ N) (m : maxTrivSubmodule R L M) :
    (maxTrivEquiv e m : N) = e ↑m :=
  rfl


@[simp]
theorem maxTrivEquiv_of_refl_eq_refl :
    maxTrivEquiv (LieModuleEquiv.refl : M ≃ₗ⁅R,L⁆ M) = LieModuleEquiv.refl := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    ⊢ Eq (LieModule.maxTrivEquiv LieModuleEquiv.refl) LieModuleEquiv.refl
  -/
  ext; simp only [coe_maxTrivEquiv_apply, LieModuleEquiv.refl_apply]
       /-
         🎉 no goals
       -/


@[simp]
theorem maxTrivEquiv_of_equiv_symm_eq_symm (e : M ≃ₗ⁅R,L⁆ N) :
    (maxTrivEquiv e).symm = maxTrivEquiv e.symm :=
  rfl


/-- A linear map between two Lie modules is a morphism of Lie modules iff the Lie algebra action
on it is trivial. -/
def maxTrivLinearMapEquivLieModuleHom : maxTrivSubmodule R L (M →ₗ[R] N) ≃ₗ[R] M →ₗ⁅R,L⁆ N where
  toFun f :=
    { toLinearMap := f.val
      map_lie' := fun {x m} => by
        /-
          R : Type u
          L : Type v
          M : Type w
          N : Type w₁
          inst✝¹⁰ : CommRing R
          inst✝⁹ : LieRing L
          inst✝⁸ : LieAlgebra R L
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          inst✝⁵ : LieRingModule L M
          inst✝⁴ : LieModule R L M
          inst✝³ : AddCommGroup N
          inst✝² : Module R N
          inst✝¹ : LieRingModule L N
          inst✝ : LieModule R L N
          f : Subtype fun x => Membership.mem (LieModule.maxTrivSubmodule R L (LinearMap …
          x : L
          m : M
          ⊢ Eq ((↑f).toFun (Bracket.bracket x m)) (Bracket.bracket x ((↑f).toFun m))
        -/
        have hf : ⁅x, f.val⁆ m = 0 := by rw [f.property x, LinearMap.zero_apply]
        /-
          R : Type u
          L : Type v
          M : Type w
          N : Type w₁
          inst✝¹⁰ : CommRing R
          inst✝⁹ : LieRing L
          inst✝⁸ : LieAlgebra R L
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          inst✝⁵ : LieRingModule L M
          inst✝⁴ : LieModule R L M
          inst✝³ : AddCommGroup N
          inst✝² : Module R N
          inst✝¹ : LieRingModule L N
          inst✝ : LieModule R L N
          f : Subtype fun x => Membership.mem (LieModule.maxTrivSubmodule R L (LinearMap …
          x : L
          m : M
          hf : Eq ((Bracket.bracket x ↑f) m) 0
          ⊢ Eq ((↑f).toFun (Bracket.bracket x m)) (Bracket.bracket x ((↑f).toFun m))
        -/
        rw [LieHom.lie_apply, sub_eq_zero, ← LinearMap.toFun_eq_coe] at hf; exact hf.symm}
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
                     /-
                       R : Type u
                       L : Type v
                       M : Type w
                       N : Type w₁
                       inst✝¹⁰ : CommRing R
                       inst✝⁹ : LieRing L
                       inst✝⁸ : LieAlgebra R L
                       inst✝⁷ : AddCommGroup M
                       inst✝⁶ : Module R M
                       inst✝⁵ : LieRingModule L M
                       inst✝⁴ : LieModule R L M
                       inst✝³ : AddCommGroup N
                       inst✝² : Module R N
                       inst✝¹ : LieRingModule L N
                       inst✝ : LieModule R L N
                       f g : Subtype fun x => Membership.mem (LieModule.maxTrivSubmodule R L (LinearM …
                       ⊢ Eq ((fun f => { toLinearMap := ↑f, map_lie' := ⋯ }) (HAdd.hAdd f g)) (HAdd.h …
                     -/
  map_add' f g := by ext; simp
                          /-
                            🎉 no goals
                          -/
                      /-
                        R : Type u
                        L : Type v
                        M : Type w
                        N : Type w₁
                        inst✝¹⁰ : CommRing R
                        inst✝⁹ : LieRing L
                        inst✝⁸ : LieAlgebra R L
                        inst✝⁷ : AddCommGroup M
                        inst✝⁶ : Module R M
                        inst✝⁵ : LieRingModule L M
                        inst✝⁴ : LieModule R L M
                        inst✝³ : AddCommGroup N
                        inst✝² : Module R N
                        inst✝¹ : LieRingModule L N
                        inst✝ : LieModule R L N
                        F : R
                        G : Subtype fun x => Membership.mem (LieModule.maxTrivSubmodule R L (LinearMap …
                        ⊢ Eq ({ toFun := fun f => { toLinearMap := ↑f, map_lie' := ⋯ }, map_add' := ⋯  …
                      -/
  map_smul' F G := by ext; simp
                           /-
                             🎉 no goals
                           -/
                              /-
                                R : Type u
                                L : Type v
                                M : Type w
                                N : Type w₁
                                inst✝¹⁰ : CommRing R
                                inst✝⁹ : LieRing L
                                inst✝⁸ : LieAlgebra R L
                                inst✝⁷ : AddCommGroup M
                                inst✝⁶ : Module R M
                                inst✝⁵ : LieRingModule L M
                                inst✝⁴ : LieModule R L M
                                inst✝³ : AddCommGroup N
                                inst✝² : Module R N
                                inst✝¹ : LieRingModule L N
                                inst✝ : LieModule R L N
                                F : LieModuleHom R L M N
                                x : L
                                ⊢ Eq (Bracket.bracket x ↑F) 0
                              -/
  invFun F := ⟨F, fun x => by ext; simp⟩
                                   /-
                                     🎉 no goals
                                   -/
                   /-
                     R : Type u
                     L : Type v
                     M : Type w
                     N : Type w₁
                     inst✝¹⁰ : CommRing R
                     inst✝⁹ : LieRing L
                     inst✝⁸ : LieAlgebra R L
                     inst✝⁷ : AddCommGroup M
                     inst✝⁶ : Module R M
                     inst✝⁵ : LieRingModule L M
                     inst✝⁴ : LieModule R L M
                     inst✝³ : AddCommGroup N
                     inst✝² : Module R N
                     inst✝¹ : LieRingModule L N
                     inst✝ : LieModule R L N
                     f : Subtype fun x => Membership.mem (LieModule.maxTrivSubmodule R L (LinearMap …
                     ⊢ Eq ((fun F => ⟨↑F, ⋯⟩) ({ toFun := fun f => { toLinearMap := ↑f, map_lie' := …
                   -/
  left_inv f := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      R : Type u
                      L : Type v
                      M : Type w
                      N : Type w₁
                      inst✝¹⁰ : CommRing R
                      inst✝⁹ : LieRing L
                      inst✝⁸ : LieAlgebra R L
                      inst✝⁷ : AddCommGroup M
                      inst✝⁶ : Module R M
                      inst✝⁵ : LieRingModule L M
                      inst✝⁴ : LieModule R L M
                      inst✝³ : AddCommGroup N
                      inst✝² : Module R N
                      inst✝¹ : LieRingModule L N
                      inst✝ : LieModule R L N
                      F : LieModuleHom R L M N
                      ⊢ Eq ({ toFun := fun f => { toLinearMap := ↑f, map_lie' := ⋯ }, map_add' := ⋯, …
                    -/
  right_inv F := by simp
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem coe_maxTrivLinearMapEquivLieModuleHom (f : maxTrivSubmodule R L (M →ₗ[R] N)) :
                                                                              /-
                                                                                R : Type u
                                                                                L : Type v
                                                                                M : Type w
                                                                                N : Type w₁
                                                                                inst✝¹⁰ : CommRing R
                                                                                inst✝⁹ : LieRing L
                                                                                inst✝⁸ : LieAlgebra R L
                                                                                inst✝⁷ : AddCommGroup M
                                                                                inst✝⁶ : Module R M
                                                                                inst✝⁵ : LieRingModule L M
                                                                                inst✝⁴ : LieModule R L M
                                                                                inst✝³ : AddCommGroup N
                                                                                inst✝² : Module R N
                                                                                inst✝¹ : LieRingModule L N
                                                                                inst✝ : LieModule R L N
                                                                                f : Subtype fun x => Membership.mem (LieModule.maxTrivSubmodule R L (LinearMap …
                                                                                ⊢ Eq ⇑(LieModule.maxTrivLinearMapEquivLieModuleHom f) ⇑↑f
                                                                              -/
    (maxTrivLinearMapEquivLieModuleHom (M := M) (N := N) f : M → N) = f := by ext; rfl
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


@[simp]
theorem coe_maxTrivLinearMapEquivLieModuleHom_symm (f : M →ₗ⁅R,L⁆ N) :
    (maxTrivLinearMapEquivLieModuleHom (M := M) (N := N) |>.symm f : M → N) = f :=
  rfl


@[simp]
theorem toLinearMap_maxTrivLinearMapEquivLieModuleHom (f : maxTrivSubmodule R L (M →ₗ[R] N)) :
    (maxTrivLinearMapEquivLieModuleHom (M := M) (N := N) f : M →ₗ[R] N) = (f : M →ₗ[R] N) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    N : Type w₁
    inst✝¹⁰ : CommRing R
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra R L
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    inst✝⁵ : LieRingModule L M
    inst✝⁴ : LieModule R L M
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : LieRingModule L N
    inst✝ : LieModule R L N
    f : Subtype fun x => Membership.mem (LieModule.maxTrivSubmodule R L (LinearMap …
    ⊢ Eq ↑(LieModule.maxTrivLinearMapEquivLieModuleHom f) ↑f
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


@[deprecated (since := "2024-12-30")]
alias coe_linearMap_maxTrivLinearMapEquivLieModuleHom :=
  toLinearMap_maxTrivLinearMapEquivLieModuleHom


@[simp]
theorem toLinearMap_maxTrivLinearMapEquivLieModuleHom_symm (f : M →ₗ⁅R,L⁆ N) :
    (maxTrivLinearMapEquivLieModuleHom (M := M) (N := N) |>.symm f : M →ₗ[R] N) = (f : M →ₗ[R] N) :=
  rfl


@[deprecated (since := "2024-12-30")]
alias coe_linearMap_maxTrivLinearMapEquivLieModuleHom_symm :=
  toLinearMap_maxTrivLinearMapEquivLieModuleHom_symm


/-- The center of a Lie algebra is the set of elements that commute with everything. It can
be viewed as the maximal trivial submodule of the Lie algebra as a Lie module over itself via the
adjoint representation. -/
abbrev center : LieIdeal R L :=
  LieModule.maxTrivSubmodule R L L


instance : IsLieAbelian (center R L) :=
  inferInstance


@[simp]
theorem ad_ker_eq_self_module_ker : (ad R L).ker = LieModule.ker R L L :=
  rfl


@[simp]
theorem self_module_ker_eq_center : LieModule.ker R L L = center R L := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    ⊢ Eq (LieModule.ker R L L) (LieAlgebra.center R L)
  -/
  ext y
  /-
    case h
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    y : L
    ⊢ Iff (Membership.mem (LieModule.ker R L L) y) (Membership.mem (LieAlgebra.cen …
  -/
  simp only [LieModule.mem_maxTrivSubmodule, LieModule.mem_ker, ← lie_skew _ y, neg_eq_zero]
  /-
    🎉 no goals
  -/


theorem abelian_of_le_center (I : LieIdeal R L) (h : I ≤ center R L) : IsLieAbelian I :=
  haveI : LieModule.IsTrivial L I := (LieModule.trivial_iff_le_maximal_trivial R L L I).mpr h
  LieIdeal.isLieAbelian_of_trivial R L I


theorem isLieAbelian_iff_center_eq_top : IsLieAbelian L ↔ center R L = ⊤ :=
  LieModule.isTrivial_iff_max_triv_eq_top R L L


lemma commute_toEnd_of_mem_center_left :
    Commute (toEnd R L M x) (toEnd R L M y) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    x : L
    hx : Membership.mem (LieAlgebra.center R L) x
    y : L
    ⊢ Commute ((LieModule.toEnd R L M) x) ((LieModule.toEnd R L M) y)
  -/
  rw [Commute.symm_iff, commute_iff_lie_eq, ← LieHom.map_lie, hx y, LieHom.map_zero]
  /-
    🎉 no goals
  -/


lemma commute_toEnd_of_mem_center_right :
    Commute (toEnd R L M y) (toEnd R L M x) :=
  (LieModule.commute_toEnd_of_mem_center_left M hx y).symm


@[simp]
theorem LieSubmodule.trivial_lie_oper_zero [LieModule.IsTrivial L M] : ⁅I, N⁆ = ⊥ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    I : LieIdeal R L
    inst✝ : LieModule.IsTrivial L M
    ⊢ Eq (Bracket.bracket I N) Bot.bot
  -/
  suffices ⁅I, N⁆ ≤ ⊥ from le_bot_iff.mp this
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    I : LieIdeal R L
    inst✝ : LieModule.IsTrivial L M
    ⊢ LE.le (Bracket.bracket I N) Bot.bot
  -/
  rw [lieIdeal_oper_eq_span, LieSubmodule.lieSpan_le]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    N : LieSubmodule R L M
    I : LieIdeal R L
    inst✝ : LieModule.IsTrivial L M
    ⊢ HasSubset.Subset (setOf fun m => Exists fun x => Exists fun n => Eq (Bracket …
  -/
  rintro m ⟨x, n, h⟩; rw [trivial_lie_zero] at h; simp [← h]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem LieSubmodule.lie_abelian_iff_lie_self_eq_bot : IsLieAbelian I ↔ ⁅I, I⁆ = ⊥ := by
  simp only [_root_.eq_bot_iff, lieIdeal_oper_eq_span, LieSubmodule.lieSpan_le,
    LieSubmodule.bot_coe, Set.subset_singleton_iff, Set.mem_setOf_eq, exists_imp]
  refine
    ⟨fun h z x y hz =>
      hz.symm.trans
        (((I : LieSubalgebra R L).coe_bracket x y).symm.trans
          ((coe_zero_iff_zero _ _).mpr (by apply h.trivial))),
      fun h => ⟨fun x y => ((I : LieSubalgebra R L).coe_zero_iff_zero _).mp (h _ x y rfl)⟩⟩


variable {I N} in
lemma lie_eq_self_of_isAtom_of_ne_bot (hN : IsAtom N) (h : ⁅I, N⁆ ≠ ⊥) : ⁅I, N⁆ = N :=
  (hN.le_iff_eq h).mp <| LieSubmodule.lie_le_right N I

-- TODO: introduce typeclass for perfect Lie algebras and use it here in the conclusion

lemma lie_eq_self_of_isAtom_of_nonabelian {R L : Type*} [CommRing R] [LieRing L] [LieAlgebra R L]
    (I : LieIdeal R L) (hI : IsAtom I) (h : ¬IsLieAbelian I) :
    ⁅I, I⁆ = I :=
  lie_eq_self_of_isAtom_of_ne_bot hI <| not_imp_not.mpr (lie_abelian_iff_lie_self_eq_bot I).mpr h



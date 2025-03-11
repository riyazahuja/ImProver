/-- A Lie submodule of a Lie module is a submodule that is closed under the Lie bracket.
This is a sufficient condition for the subset itself to form a Lie module. -/
structure LieSubmodule extends Submodule R M where
  lie_mem : ∀ {x : L} {m : M}, m ∈ carrier → ⁅x, m⁆ ∈ carrier


instance : SetLike (LieSubmodule R L M) M where
  coe s := s.carrier
                             /-
                               R : Type u
                               L : Type v
                               M : Type w
                               inst✝⁴ : CommRing R
                               inst✝³ : LieRing L
                               inst✝² : AddCommGroup M
                               inst✝¹ : Module R M
                               inst✝ : LieRingModule L M
                               N✝ N' N O : LieSubmodule R L M
                               h : Eq ((fun s => (↑s).carrier) N) ((fun s => (↑s).carrier) O)
                               ⊢ Eq N O
                             -/
  coe_injective' N O h := by cases N; cases O; congr; exact SetLike.coe_injective' h
                                                      /-
                                                        🎉 no goals
                                                      -/


instance : AddSubgroupClass (LieSubmodule R L M) M where
  add_mem {N} _ _ := N.add_mem'
  zero_mem N := N.zero_mem'
  neg_mem {N} x hx := show -x ∈ N.toSubmodule from neg_mem hx


instance instSMulMemClass : SMulMemClass (LieSubmodule R L M) R M where
  smul_mem {s} c _ h := s.smul_mem'  c h


/-- The zero module is a Lie submodule of any Lie module. -/
instance : Zero (LieSubmodule R L M) :=
  ⟨{ (0 : Submodule R M) with
                                  /-
                                    R : Type u
                                    L : Type v
                                    M : Type w
                                    inst✝⁴ : CommRing R
                                    inst✝³ : LieRing L
                                    inst✝² : AddCommGroup M
                                    inst✝¹ : Module R M
                                    inst✝ : LieRingModule L M
                                    N N' : LieSubmodule R L M
                                    x : L
                                    m : M
                                    h : Membership.mem __src✝.carrier m
                                    ⊢ Membership.mem __src✝.carrier (Bracket.bracket x m)
                                  -/
      lie_mem := fun {x m} h ↦ by rw [(Submodule.mem_bot R).1 h]; apply lie_zero }⟩
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


instance : Inhabited (LieSubmodule R L M) :=
  ⟨0⟩


instance (priority := high) coeSort : CoeSort (LieSubmodule R L M) (Type w) where
  coe N := { x : M // x ∈ N }


instance (priority := mid) coeSubmodule : CoeOut (LieSubmodule R L M) (Submodule R M) :=
  ⟨toSubmodule⟩


instance : CanLift (Submodule R M) (LieSubmodule R L M) (·)
    (fun N ↦ ∀ {x : L} {m : M}, m ∈ N → ⁅x, m⁆ ∈ N) where
  prf N hN := ⟨⟨N, hN⟩, rfl⟩


@[norm_cast]
theorem coe_toSubmodule : ((N : Submodule R M) : Set M) = N :=
  rfl

-- `simp` can prove this after `mem_toSubmodule` is added to the simp set,
-- but `dsimp` can't.

@[simp, nolint simpNF]
theorem mem_carrier {x : M} : x ∈ N.carrier ↔ x ∈ (N : Set M) :=
  Iff.rfl


theorem mem_mk_iff (S : Set M) (h₁ h₂ h₃ h₄) {x : M} :
    x ∈ (⟨⟨⟨⟨S, h₁⟩, h₂⟩, h₃⟩, h₄⟩ : LieSubmodule R L M) ↔ x ∈ S :=
  Iff.rfl


@[simp]
theorem mem_mk_iff' (p : Submodule R M) (h) {x : M} :
    x ∈ (⟨p, h⟩ : LieSubmodule R L M) ↔ x ∈ p :=
  Iff.rfl


@[simp]
theorem mem_toSubmodule {x : M} : x ∈ (N : Submodule R M) ↔ x ∈ N :=
  Iff.rfl


@[deprecated (since := "2024-12-30")] alias mem_coeSubmodule := mem_toSubmodule


theorem mem_coe {x : M} : x ∈ (N : Set M) ↔ x ∈ N :=
  Iff.rfl


@[simp]
protected theorem zero_mem : (0 : M) ∈ N :=
  zero_mem N


@[simp]
theorem mk_eq_zero {x} (h : x ∈ N) : (⟨x, h⟩ : N) = 0 ↔ x = 0 :=
  Subtype.ext_iff_val


@[simp]
theorem coe_toSet_mk (S : Set M) (h₁ h₂ h₃ h₄) :
    ((⟨⟨⟨⟨S, h₁⟩, h₂⟩, h₃⟩, h₄⟩ : LieSubmodule R L M) : Set M) = S :=
  rfl


theorem toSubmodule_mk (p : Submodule R M) (h) :
                                                                               /-
                                                                                 R : Type u
                                                                                 L : Type v
                                                                                 M : Type w
                                                                                 inst✝⁴ : CommRing R
                                                                                 inst✝³ : LieRing L
                                                                                 inst✝² : AddCommGroup M
                                                                                 inst✝¹ : Module R M
                                                                                 inst✝ : LieRingModule L M
                                                                                 p : Submodule R M
                                                                                 h : ∀ {x : L} {m : M}, Membership.mem p.carrier m → Membership.mem p.carrier ( …
                                                                                 ⊢ Eq (↑{ toSubmodule := p, lie_mem := h }) p
                                                                               -/
    (({ p with lie_mem := h } : LieSubmodule R L M) : Submodule R M) = p := by cases p; rfl
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


@[deprecated (since := "2024-12-30")] alias coe_toSubmodule_mk := toSubmodule_mk


theorem toSubmodule_injective :
    Function.Injective (toSubmodule : LieSubmodule R L M → Submodule R M) := fun x y h ↦ by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    x y : LieSubmodule R L M
    h : Eq ↑x ↑y
    ⊢ Eq x y
  -/
  cases x; cases y; congr
                    /-
                      🎉 no goals
                    -/


@[deprecated (since := "2024-12-30")] alias coeSubmodule_injective := toSubmodule_injective


@[ext]
theorem ext (h : ∀ m, m ∈ N ↔ m ∈ N') : N = N' :=
  SetLike.ext h


@[simp]
theorem toSubmodule_inj : (N : Submodule R M) = (N' : Submodule R M) ↔ N = N' :=
  toSubmodule_injective.eq_iff


@[deprecated (since := "2024-12-30")] alias coe_toSubmodule_inj := toSubmodule_inj


@[deprecated (since := "2024-12-29")] alias toSubmodule_eq_iff := toSubmodule_inj


/-- Copy of a `LieSubmodule` with a new `carrier` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (s : Set M) (hs : s = ↑N) : LieSubmodule R L M where
  carrier := s
  -- Porting note: all the proofs below were in term mode
                  /-
                    R : Type u
                    L : Type v
                    M : Type w
                    inst✝⁴ : CommRing R
                    inst✝³ : LieRing L
                    inst✝² : AddCommGroup M
                    inst✝¹ : Module R M
                    inst✝ : LieRingModule L M
                    N N' : LieSubmodule R L M
                    s : Set M
                    hs : Eq s ↑N
                    ⊢ Membership.mem { carrier := s, add_mem' := ⋯ }.carrier 0
                  -/
                     /-
                       R : Type u
                       L : Type v
                       M : Type w
                       inst✝⁴ : CommRing R
                       inst✝³ : LieRing L
                       inst✝² : AddCommGroup M
                       inst✝¹ : Module R M
                       inst✝ : LieRingModule L M
                       N N' : LieSubmodule R L M
                       s : Set M
                       hs : Eq s ↑N
                       a✝ b✝ : M
                       x : Membership.mem s a✝
                       y : Membership.mem s b✝
                       ⊢ Membership.mem s (HAdd.hAdd a✝ b✝)
                     -/
  zero_mem' := by exact hs.symm ▸ N.zero_mem'
                                       /-
                                         🎉 no goals
                                       -/
                  /-
                    🎉 no goals
                  -/
  add_mem' x y := by rw [hs] at x y ⊢; exact N.add_mem' x y
                  /-
                    R : Type u
                    L : Type v
                    M : Type w
                    inst✝⁴ : CommRing R
                    inst✝³ : LieRing L
                    inst✝² : AddCommGroup M
                    inst✝¹ : Module R M
                    inst✝ : LieRingModule L M
                    N N' : LieSubmodule R L M
                    s : Set M
                    hs : Eq s ↑N
                    ⊢ ∀ (c : R) {x : M}, Membership.mem { carrier := s, add_mem' := ⋯, zero_mem' : …
                  -/
  smul_mem' := by exact hs.symm ▸ N.smul_mem'
                  /-
                    🎉 no goals
                  -/
                /-
                  R : Type u
                  L : Type v
                  M : Type w
                  inst✝⁴ : CommRing R
                  inst✝³ : LieRing L
                  inst✝² : AddCommGroup M
                  inst✝¹ : Module R M
                  inst✝ : LieRingModule L M
                  N N' : LieSubmodule R L M
                  s : Set M
                  hs : Eq s ↑N
                  ⊢ ∀ {x : L} {m : M}, Membership.mem { carrier := s, add_mem' := ⋯, zero_mem' : …
                -/
  lie_mem := by exact hs.symm ▸ N.lie_mem
                /-
                  🎉 no goals
                -/


@[simp]
theorem coe_copy (S : LieSubmodule R L M) (s : Set M) (hs : s = ↑S) : (S.copy s hs : Set M) = s :=
  rfl


theorem copy_eq (S : LieSubmodule R L M) (s : Set M) (hs : s = ↑S) : S.copy s hs = S :=
  SetLike.coe_injective hs


instance : LieRingModule L N where
  bracket (x : L) (m : N) := ⟨⁅x, m.val⁆, N.lie_mem m.property⟩
                /-
                  R : Type u
                  L : Type v
                  M : Type w
                  inst✝⁴ : CommRing R
                  inst✝³ : LieRing L
                  inst✝² : AddCommGroup M
                  inst✝¹ : Module R M
                  inst✝ : LieRingModule L M
                  N N' : LieSubmodule R L M
                  ⊢ ∀ (x y : L) (m : Subtype fun x => Membership.mem N x), Eq (Bracket.bracket ( …
                -/
  add_lie := by intro x y m; apply SetCoe.ext; apply add_lie
                                               /-
                                                 🎉 no goals
                                               -/
                /-
                  R : Type u
                  L : Type v
                  M : Type w
                  inst✝⁴ : CommRing R
                  inst✝³ : LieRing L
                  inst✝² : AddCommGroup M
                  inst✝¹ : Module R M
                  inst✝ : LieRingModule L M
                  N N' : LieSubmodule R L M
                  ⊢ ∀ (x : L) (m n : Subtype fun x => Membership.mem N x), Eq (Bracket.bracket x …
                -/
  lie_add := by intro x m n; apply SetCoe.ext; apply lie_add
                                               /-
                                                 🎉 no goals
                                               -/
                    /-
                      R : Type u
                      L : Type v
                      M : Type w
                      inst✝⁴ : CommRing R
                      inst✝³ : LieRing L
                      inst✝² : AddCommGroup M
                      inst✝¹ : Module R M
                      inst✝ : LieRingModule L M
                      N N' : LieSubmodule R L M
                      ⊢ ∀ (x y : L) (m : Subtype fun x => Membership.mem N x), Eq (Bracket.bracket x …
                    -/
  leibniz_lie := by intro x y m; apply SetCoe.ext; apply leibniz_lie
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp, norm_cast]
theorem coe_zero : ((0 : N) : M) = (0 : M) :=
  rfl


@[simp, norm_cast]
theorem coe_add (m m' : N) : (↑(m + m') : M) = (m : M) + (m' : M) :=
  rfl


@[simp, norm_cast]
theorem coe_neg (m : N) : (↑(-m) : M) = -(m : M) :=
  rfl


@[simp, norm_cast]
theorem coe_sub (m m' : N) : (↑(m - m') : M) = (m : M) - (m' : M) :=
  rfl


@[simp, norm_cast]
theorem coe_smul (t : R) (m : N) : (↑(t • m) : M) = t • (m : M) :=
  rfl


@[simp, norm_cast]
theorem coe_bracket (x : L) (m : N) :
    (↑⁅x, m⁆ : M) = ⁅x, ↑m⁆ :=
  rfl

-- Copying instances from `Submodule` for correct discrimination keys

instance [IsNoetherian R M] (N : LieSubmodule R L M) : IsNoetherian R N :=
  inferInstanceAs <| IsNoetherian R N.toSubmodule


instance [IsArtinian R M] (N : LieSubmodule R L M) : IsArtinian R N :=
  inferInstanceAs <| IsArtinian R N.toSubmodule


instance [NoZeroSMulDivisors R M] : NoZeroSMulDivisors R N :=
  inferInstanceAs <| NoZeroSMulDivisors R N.toSubmodule


instance instLieModule : LieModule R L N where
                 /-
                   R : Type u
                   L : Type v
                   M : Type w
                   inst✝⁶ : CommRing R
                   inst✝⁵ : LieRing L
                   inst✝⁴ : AddCommGroup M
                   inst✝³ : Module R M
                   inst✝² : LieRingModule L M
                   N N' : LieSubmodule R L M
                   inst✝¹ : LieAlgebra R L
                   inst✝ : LieModule R L M
                   ⊢ ∀ (t : R) (x : L) (m : Subtype fun x => Membership.mem N x), Eq (Bracket.bra …
                 -/
                 /-
                   R : Type u
                   L : Type v
                   M : Type w
                   inst✝⁶ : CommRing R
                   inst✝⁵ : LieRing L
                   inst✝⁴ : AddCommGroup M
                   inst✝³ : Module R M
                   inst✝² : LieRingModule L M
                   N N' : LieSubmodule R L M
                   inst✝¹ : LieAlgebra R L
                   inst✝ : LieModule R L M
                   ⊢ ∀ (t : R) (x : L) (m : Subtype fun x => Membership.mem N x), Eq (Bracket.bra …
                 -/
  lie_smul := by intro t x y; apply SetCoe.ext; apply lie_smul
                                                /-
                                                  🎉 no goals
                                                -/
                                                /-
                                                  🎉 no goals
                                                -/
  smul_lie := by intro t x y; apply SetCoe.ext; apply smul_lie


instance [Subsingleton M] : Unique (LieSubmodule R L M) :=
  ⟨⟨0⟩, fun _ ↦ (toSubmodule_inj _ _).mp (Subsingleton.elim _ _)⟩


/-- An ideal of a Lie algebra is a Lie submodule of the Lie algebra as a Lie module over itself. -/
abbrev LieIdeal :=
  LieSubmodule R L L


theorem lie_mem_right (I : LieIdeal R L) (x y : L) (h : y ∈ I) : ⁅x, y⁆ ∈ I :=
  I.lie_mem h


theorem lie_mem_left (I : LieIdeal R L) (x y : L) (h : x ∈ I) : ⁅x, y⁆ ∈ I := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    x y : L
    h : Membership.mem I x
    ⊢ Membership.mem I (Bracket.bracket x y)
  -/
  rw [← lie_skew, ← neg_lie]; apply lie_mem_right; assumption
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- An ideal of a Lie algebra is a Lie subalgebra. -/
def LieIdeal.toLieSubalgebra (I : LieIdeal R L) : LieSubalgebra R L :=
                                      /-
                                        R : Type u
                                        L : Type v
                                        M : Type w
                                        inst✝⁶ : CommRing R
                                        inst✝⁵ : LieRing L
                                        inst✝⁴ : AddCommGroup M
                                        inst✝³ : Module R M
                                        inst✝² : LieRingModule L M
                                        inst✝¹ : LieAlgebra R L
                                        inst✝ : LieModule R L M
                                        I : LieIdeal R L
                                        ⊢ ∀ {x y : L}, Membership.mem __src✝.carrier x → Membership.mem __src✝.carrier …
                                      -/
  { I.toSubmodule with lie_mem' := by intro x y _ hy; apply lie_mem_right; exact hy }
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[deprecated (since := "2025-01-02")] alias lieIdealSubalgebra := LieIdeal.toLieSubalgebra


instance : Coe (LieIdeal R L) (LieSubalgebra R L) :=
  ⟨LieIdeal.toLieSubalgebra R L⟩


@[simp]
theorem LieIdeal.coe_toLieSubalgebra (I : LieIdeal R L) : ((I : LieSubalgebra R L) : Set L) = I :=
  rfl


@[deprecated (since := "2024-12-30")]
alias LieIdeal.coe_toSubalgebra := LieIdeal.coe_toLieSubalgebra


@[simp]
theorem LieIdeal.toLieSubalgebra_toSubmodule (I : LieIdeal R L) :
    ((I : LieSubalgebra R L) : Submodule R L) = LieSubmodule.toSubmodule I :=
  rfl


@[deprecated (since := "2025-01-02")]
alias LieIdeal.coe_toLieSubalgebra_toSubmodule := LieIdeal.toLieSubalgebra_toSubmodule


@[deprecated (since := "2024-12-30")]
alias LieIdeal.coe_to_lieSubalgebra_to_submodule := LieIdeal.toLieSubalgebra_toSubmodule


/-- An ideal of `L` is a Lie subalgebra of `L`, so it is a Lie ring. -/
instance LieIdeal.lieRing (I : LieIdeal R L) : LieRing I :=
  LieSubalgebra.lieRing R L ↑I


/-- Transfer the `LieAlgebra` instance from the coercion `LieIdeal → LieSubalgebra`. -/
instance LieIdeal.lieAlgebra (I : LieIdeal R L) : LieAlgebra R I :=
  LieSubalgebra.lieAlgebra R L ↑I


/-- Transfer the `LieRingModule` instance from the coercion `LieIdeal → LieSubalgebra`. -/
instance LieIdeal.lieRingModule {R L : Type*} [CommRing R] [LieRing L] [LieAlgebra R L]
    (I : LieIdeal R L) [LieRingModule L M] : LieRingModule I M :=
  LieSubalgebra.lieRingModule (I : LieSubalgebra R L)


@[simp]
theorem LieIdeal.coe_bracket_of_module {R L : Type*} [CommRing R] [LieRing L] [LieAlgebra R L]
    (I : LieIdeal R L) [LieRingModule L M] (x : I) (m : M) :
    ⁅x, m⁆ = ⁅(↑x : L), m⁆ :=
  LieSubalgebra.coe_bracket_of_module (I : LieSubalgebra R L) x m


/-- Transfer the `LieModule` instance from the coercion `LieIdeal → LieSubalgebra`. -/
instance LieIdeal.lieModule (I : LieIdeal R L) : LieModule R I M :=
  LieSubalgebra.lieModule (I : LieSubalgebra R L)


instance (I : LieIdeal R L) : IsLieTower I L M where
  leibniz_lie x y m := leibniz_lie x.val y m


instance (I : LieIdeal R L) : IsLieTower L I M where
  leibniz_lie x y m := leibniz_lie x y.val m


theorem Submodule.exists_lieSubmodule_coe_eq_iff (p : Submodule R M) :
    (∃ N : LieSubmodule R L M, ↑N = p) ↔ ∀ (x : L) (m : M), m ∈ p → ⁅x, m⁆ ∈ p := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    p : Submodule R M
    ⊢ Iff (Exists fun N => Eq (↑N) p) (∀ (x : L) (m : M), Membership.mem p m → Mem …
  -/
  constructor
    /-
      case mp
      R : Type u
      L : Type v
      M : Type w
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      p : Submodule R M
      ⊢ (Exists fun N => Eq (↑N) p) → ∀ (x : L) (m : M), Membership.mem p m → Member …
    -/
  · rintro ⟨N, rfl⟩ _ _; exact N.lie_mem
                         /-
                           🎉 no goals
                         -/
    /-
      case mpr
      R : Type u
      L : Type v
      M : Type w
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      p : Submodule R M
      ⊢ (∀ (x : L) (m : M), Membership.mem p m → Membership.mem p (Bracket.bracket x …
    -/
  · intro h; use { p with lie_mem := @h }
             /-
               🎉 no goals
             -/


/-- Given a Lie subalgebra `K ⊆ L`, if we view `L` as a `K`-module by restriction, it contains
a distinguished Lie submodule for the action of `K`, namely `K` itself. -/
def toLieSubmodule : LieSubmodule R K L :=
  { (K : Submodule R L) with lie_mem := fun {x _} hy ↦ K.lie_mem x.property hy }


@[simp]
theorem coe_toLieSubmodule : (K.toLieSubmodule : Submodule R L) = K := rfl


@[simp]
theorem mem_toLieSubmodule (x : L) : x ∈ K.toLieSubmodule ↔ x ∈ K :=
  Iff.rfl


theorem exists_lieIdeal_coe_eq_iff :
    (∃ I : LieIdeal R L, ↑I = K) ↔ ∀ x y : L, y ∈ K → ⁅x, y⁆ ∈ K := by
  simp only [← toSubmodule_inj, LieIdeal.toLieSubalgebra_toSubmodule,
    Submodule.exists_lieSubmodule_coe_eq_iff L]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K : LieSubalgebra R L
    ⊢ Iff (∀ (x m : L), Membership.mem K.toSubmodule m → Membership.mem K.toSubmod …
  -/
  exact Iff.rfl
  /-
    🎉 no goals
  -/


theorem exists_nested_lieIdeal_coe_eq_iff {K' : LieSubalgebra R L} (h : K ≤ K') :
    (∃ I : LieIdeal R K', ↑I = ofLe h) ↔ ∀ x y : L, x ∈ K' → y ∈ K → ⁅x, y⁆ ∈ K := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K K' : LieSubalgebra R L
    h : LE.le K K'
    ⊢ Iff (Exists fun I => Eq (LieIdeal.toLieSubalgebra R (Subtype fun x => Member …
  -/
  simp only [exists_lieIdeal_coe_eq_iff, coe_bracket, mem_ofLe]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    K K' : LieSubalgebra R L
    h : LE.le K K'
    ⊢ Iff (∀ (x y : Subtype fun x => Membership.mem K' x), Membership.mem K ↑y → M …
  -/
  constructor
    /-
      case mp
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      K K' : LieSubalgebra R L
      h : LE.le K K'
      ⊢ (∀ (x y : Subtype fun x => Membership.mem K' x), Membership.mem K ↑y → Membe …
    -/
  · intro h' x y hx hy; exact h' ⟨x, hx⟩ ⟨y, h hy⟩ hy
                        /-
                          🎉 no goals
                        -/
    /-
      case mpr
      R : Type u
      L : Type v
      inst✝² : CommRing R
      inst✝¹ : LieRing L
      inst✝ : LieAlgebra R L
      K K' : LieSubalgebra R L
      h : LE.le K K'
      ⊢ (∀ (x y : L), Membership.mem K' x → Membership.mem K y → Membership.mem K (B …
    -/
  · rintro h' ⟨x, hx⟩ ⟨y, hy⟩ hy'; exact h' x y hx hy'
                                   /-
                                     🎉 no goals
                                   -/


theorem coe_injective : Function.Injective ((↑) : LieSubmodule R L M → Set M) :=
  SetLike.coe_injective


@[simp, norm_cast]
theorem toSubmodule_le_toSubmodule : (N : Submodule R M) ≤ N' ↔ N ≤ N' :=
  Iff.rfl


@[deprecated (since := "2024-12-30")]
alias coeSubmodule_le_coeSubmodule := toSubmodule_le_toSubmodule


instance : Bot (LieSubmodule R L M) :=
  ⟨0⟩


instance instUniqueBot : Unique (⊥ : LieSubmodule R L M) :=
  inferInstanceAs <| Unique (⊥ : Submodule R M)


@[simp]
theorem bot_coe : ((⊥ : LieSubmodule R L M) : Set M) = {0} :=
  rfl


@[simp]
theorem bot_toSubmodule : ((⊥ : LieSubmodule R L M) : Submodule R M) = ⊥ :=
  rfl


@[deprecated (since := "2024-12-30")] alias bot_coeSubmodule := bot_toSubmodule


@[simp]
theorem toSubmodule_eq_bot : (N : Submodule R M) = ⊥ ↔ N = ⊥ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N : LieSubmodule R L M
    ⊢ Iff (Eq (↑N) Bot.bot) (Eq N Bot.bot)
  -/
  rw [← toSubmodule_inj, bot_toSubmodule]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-30")] alias coeSubmodule_eq_bot_iff := toSubmodule_eq_bot


@[simp] theorem mk_eq_bot_iff {N : Submodule R M} {h} :
    (⟨N, h⟩ : LieSubmodule R L M) = ⊥ ↔ N = ⊥ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N : Submodule R M
    h : ∀ {x : L} {m : M}, Membership.mem N.carrier m → Membership.mem N.carrier ( …
    ⊢ Iff (Eq { toSubmodule := N, lie_mem := h } Bot.bot) (Eq N Bot.bot)
  -/
  rw [← toSubmodule_inj, bot_toSubmodule]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_bot (x : M) : x ∈ (⊥ : LieSubmodule R L M) ↔ x = 0 :=
  mem_singleton_iff


instance : Top (LieSubmodule R L M) :=
  ⟨{ (⊤ : Submodule R M) with lie_mem := fun {x m} _ ↦ mem_univ ⁅x, m⁆ }⟩


@[simp]
theorem top_coe : ((⊤ : LieSubmodule R L M) : Set M) = univ :=
  rfl


@[simp]
theorem top_toSubmodule : ((⊤ : LieSubmodule R L M) : Submodule R M) = ⊤ :=
  rfl


@[deprecated (since := "2024-12-30")] alias top_coeSubmodule := top_toSubmodule


@[simp]
theorem toSubmodule_eq_top : (N : Submodule R M) = ⊤ ↔ N = ⊤ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N : LieSubmodule R L M
    ⊢ Iff (Eq (↑N) Top.top) (Eq N Top.top)
  -/
  rw [← toSubmodule_inj, top_toSubmodule]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-30")] alias coeSubmodule_eq_top_iff := toSubmodule_eq_top


@[simp] theorem mk_eq_top_iff {N : Submodule R M} {h} :
    (⟨N, h⟩ : LieSubmodule R L M) = ⊤ ↔ N = ⊤ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N : Submodule R M
    h : ∀ {x : L} {m : M}, Membership.mem N.carrier m → Membership.mem N.carrier ( …
    ⊢ Iff (Eq { toSubmodule := N, lie_mem := h } Top.top) (Eq N Top.top)
  -/
  rw [← toSubmodule_inj, top_toSubmodule]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_top (x : M) : x ∈ (⊤ : LieSubmodule R L M) :=
  mem_univ x


instance : Min (LieSubmodule R L M) :=
  ⟨fun N N' ↦
    { (N ⊓ N' : Submodule R M) with
      lie_mem := fun h ↦ mem_inter (N.lie_mem h.1) (N'.lie_mem h.2) }⟩


instance : InfSet (LieSubmodule R L M) :=
  ⟨fun S ↦
    { toSubmodule := sInf {(s : Submodule R M) | s ∈ S}
      lie_mem := fun {x m} h ↦ by
        simp only [Submodule.mem_carrier, mem_iInter, Submodule.sInf_coe, mem_setOf_eq,
          forall_apply_eq_imp_iff₂, forall_exists_index, and_imp] at h ⊢
        /-
          R : Type u
          L : Type v
          M : Type w
          inst✝⁴ : CommRing R
          inst✝³ : LieRing L
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : LieRingModule L M
          N N' : LieSubmodule R L M
          S : Set (LieSubmodule R L M)
          x : L
          m : M
          h : ∀ (a : LieSubmodule R L M), Membership.mem S a → Membership.mem (↑↑a) m
          ⊢ ∀ (a : LieSubmodule R L M), Membership.mem S a → Membership.mem (↑↑a) (Brack …
        -/
        intro N hN; apply N.lie_mem (h N hN) }⟩
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem inf_coe : (↑(N ⊓ N') : Set M) = ↑N ∩ ↑N' :=
  rfl


@[norm_cast, simp]
theorem inf_toSubmodule :
    (↑(N ⊓ N') : Submodule R M) = (N : Submodule R M) ⊓ (N' : Submodule R M) :=
  rfl


@[deprecated (since := "2024-12-30")] alias inf_coe_toSubmodule := inf_toSubmodule


@[simp]
theorem sInf_toSubmodule (S : Set (LieSubmodule R L M)) :
    (↑(sInf S) : Submodule R M) = sInf {(s : Submodule R M) | s ∈ S} :=
  rfl


@[deprecated (since := "2024-12-30")] alias sInf_coe_toSubmodule := sInf_toSubmodule


theorem sInf_toSubmodule_eq_iInf (S : Set (LieSubmodule R L M)) :
    (↑(sInf S) : Submodule R M) = ⨅ N ∈ S, (N : Submodule R M) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    S : Set (LieSubmodule R L M)
    ⊢ Eq (↑(InfSet.sInf S)) (iInf fun N => iInf fun h => ↑N)
  -/
  rw [sInf_toSubmodule, ← Set.image, sInf_image]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-30")] alias sInf_coe_toSubmodule' := sInf_toSubmodule_eq_iInf


@[simp]
theorem iInf_toSubmodule {ι} (p : ι → LieSubmodule R L M) :
    (↑(⨅ i, p i) : Submodule R M) = ⨅ i, (p i : Submodule R M) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    ι : Sort u_1
    p : ι → LieSubmodule R L M
    ⊢ Eq (↑(iInf fun i => p i)) (iInf fun i => ↑(p i))
  -/
  rw [iInf, sInf_toSubmodule]; ext; simp
                                    /-
                                      🎉 no goals
                                    -/


@[deprecated (since := "2024-12-30")] alias iInf_coe_toSubmodule := iInf_toSubmodule


@[simp]
theorem sInf_coe (S : Set (LieSubmodule R L M)) : (↑(sInf S) : Set M) = ⋂ s ∈ S, (s : Set M) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    S : Set (LieSubmodule R L M)
    ⊢ Eq (↑(InfSet.sInf S)) (Set.iInter fun s => Set.iInter fun h => ↑s)
  -/
  rw [← LieSubmodule.coe_toSubmodule, sInf_toSubmodule, Submodule.sInf_coe]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    S : Set (LieSubmodule R L M)
    ⊢ Eq (Set.iInter fun p => Set.iInter fun h => ↑p) (Set.iInter fun s => Set.iIn …
  -/
  ext m
  simp only [mem_iInter, mem_setOf_eq, forall_apply_eq_imp_iff₂, exists_imp,
    and_imp, SetLike.mem_coe, mem_toSubmodule]


@[simp]
theorem iInf_coe {ι} (p : ι → LieSubmodule R L M) : (↑(⨅ i, p i) : Set M) = ⋂ i, ↑(p i) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    ι : Sort u_1
    p : ι → LieSubmodule R L M
    ⊢ Eq (↑(iInf fun i => p i)) (Set.iInter fun i => ↑(p i))
  -/
  rw [iInf, sInf_coe]; simp only [Set.mem_range, Set.iInter_exists, Set.iInter_iInter_eq']
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem mem_iInf {ι} (p : ι → LieSubmodule R L M) {x} : (x ∈ ⨅ i, p i) ↔ ∀ i, x ∈ p i := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    ι : Sort u_1
    p : ι → LieSubmodule R L M
    x : M
    ⊢ Iff (Membership.mem (iInf fun i => p i) x) (∀ (i : ι), Membership.mem (p i) x)
  -/
  rw [← SetLike.mem_coe, iInf_coe, Set.mem_iInter]; rfl
                                                    /-
                                                      🎉 no goals
                                                    -/


instance : Max (LieSubmodule R L M) where
  max N N' :=
    { toSubmodule := (N : Submodule R M) ⊔ (N' : Submodule R M)
      lie_mem := by
        /-
          R : Type u
          L : Type v
          M : Type w
          inst✝⁴ : CommRing R
          inst✝³ : LieRing L
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : LieRingModule L M
          N✝ N'✝ N N' : LieSubmodule R L M
          ⊢ ∀ {x : L} {m : M}, Membership.mem (Max.max ↑N ↑N').carrier m → Membership.me …
        -/
        rintro x m (hm : m ∈ (N : Submodule R M) ⊔ (N' : Submodule R M))
        /-
          R : Type u
          L : Type v
          M : Type w
          inst✝⁴ : CommRing R
          inst✝³ : LieRing L
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : LieRingModule L M
          N✝ N'✝ N N' : LieSubmodule R L M
          x : L
          m : M
          hm : Membership.mem (Max.max ↑N ↑N') m
          ⊢ Membership.mem (Max.max ↑N ↑N').carrier (Bracket.bracket x m)
        -/
        change ⁅x, m⁆ ∈ (N : Submodule R M) ⊔ (N' : Submodule R M)
        /-
          R : Type u
          L : Type v
          M : Type w
          inst✝⁴ : CommRing R
          inst✝³ : LieRing L
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : LieRingModule L M
          N✝ N'✝ N N' : LieSubmodule R L M
          x : L
          m : M
          hm : Membership.mem (Max.max ↑N ↑N') m
          ⊢ Membership.mem (Max.max ↑N ↑N') (Bracket.bracket x m)
        -/
        rw [Submodule.mem_sup] at hm ⊢
        /-
          R : Type u
          L : Type v
          M : Type w
          inst✝⁴ : CommRing R
          inst✝³ : LieRing L
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : LieRingModule L M
          N✝ N'✝ N N' : LieSubmodule R L M
          x : L
          m : M
          hm : Exists fun y => And (Membership.mem (↑N) y) (Exists fun z => And (Members …
          ⊢ Exists fun y => And (Membership.mem (↑N) y) (Exists fun z => And (Membership …
        -/
        obtain ⟨y, hy, z, hz, rfl⟩ := hm
        /-
          case intro.intro.intro.intro
          R : Type u
          L : Type v
          M : Type w
          inst✝⁴ : CommRing R
          inst✝³ : LieRing L
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : LieRingModule L M
          N✝ N'✝ N N' : LieSubmodule R L M
          x : L
          y : M
          hy : Membership.mem (↑N) y
          z : M
          hz : Membership.mem (↑N') z
          ⊢ Exists fun y_1 => And (Membership.mem (↑N) y_1) (Exists fun z_1 => And (Memb …
        -/
        exact ⟨⁅x, y⁆, N.lie_mem hy, ⁅x, z⁆, N'.lie_mem hz, (lie_add _ _ _).symm⟩ }
        /-
          🎉 no goals
        -/


instance : SupSet (LieSubmodule R L M) where
  sSup S :=
    { toSubmodule := sSup {(p : Submodule R M) | p ∈ S}
      lie_mem := by
        /-
          R : Type u
          L : Type v
          M : Type w
          inst✝⁴ : CommRing R
          inst✝³ : LieRing L
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : LieRingModule L M
          N N' : LieSubmodule R L M
          S : Set (LieSubmodule R L M)
          ⊢ ∀ {x : L} {m : M}, Membership.mem (SupSet.sSup (setOf fun x => Exists fun p  …
        -/
        intro x m (hm : m ∈ sSup {(p : Submodule R M) | p ∈ S})
        /-
          R : Type u
          L : Type v
          M : Type w
          inst✝⁴ : CommRing R
          inst✝³ : LieRing L
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : LieRingModule L M
          N N' : LieSubmodule R L M
          S : Set (LieSubmodule R L M)
          x : L
          m : M
          hm : Membership.mem (SupSet.sSup (setOf fun x => Exists fun p => And (Membersh …
          ⊢ Membership.mem (SupSet.sSup (setOf fun x => Exists fun p => And (Membership. …
        -/
        change ⁅x, m⁆ ∈ sSup {(p : Submodule R M) | p ∈ S}
        /-
          R : Type u
          L : Type v
          M : Type w
          inst✝⁴ : CommRing R
          inst✝³ : LieRing L
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : LieRingModule L M
          N N' : LieSubmodule R L M
          S : Set (LieSubmodule R L M)
          x : L
          m : M
          hm : Membership.mem (SupSet.sSup (setOf fun x => Exists fun p => And (Membersh …
          ⊢ Membership.mem (SupSet.sSup (setOf fun x => Exists fun p => And (Membership. …
        -/
        obtain ⟨s, hs, hsm⟩ := Submodule.mem_sSup_iff_exists_finset.mp hm
        /-
          case intro.intro
          R : Type u
          L : Type v
          M : Type w
          inst✝⁴ : CommRing R
          inst✝³ : LieRing L
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : LieRingModule L M
          N N' : LieSubmodule R L M
          S : Set (LieSubmodule R L M)
          x : L
          m : M
          hm : Membership.mem (SupSet.sSup (setOf fun x => Exists fun p => And (Membersh …
          s : Finset (Submodule R M)
          hs : HasSubset.Subset (↑s) (setOf fun x => Exists fun p => And (Membership.mem …
          hsm : Membership.mem (iSup fun i => iSup fun h => i) m
          ⊢ Membership.mem (SupSet.sSup (setOf fun x => Exists fun p => And (Membership. …
        -/
        clear hm
        classical
        induction s using Finset.induction_on generalizing m with
        | empty =>
          replace hsm : m = 0 := by simpa using hsm
          simp [hsm]
        | insert hqt ih =>
          rename_i q t
          rw [Finset.iSup_insert] at hsm
          obtain ⟨m', hm', u, hu, rfl⟩ := Submodule.mem_sup.mp hsm
          rw [lie_add]
          refine add_mem ?_ (ih (Subset.trans (by simp) hs) hu)
          obtain ⟨p, hp, rfl⟩ : ∃ p ∈ S, ↑p = q := hs (Finset.mem_insert_self q t)
          suffices p ≤ sSup {(p : Submodule R M) | p ∈ S} by exact this (p.lie_mem hm')
          exact le_sSup ⟨p, hp, rfl⟩ }


@[norm_cast, simp]
theorem sup_toSubmodule :
    (↑(N ⊔ N') : Submodule R M) = (N : Submodule R M) ⊔ (N' : Submodule R M) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N N' : LieSubmodule R L M
    ⊢ Eq (↑(Max.max N N')) (Max.max ↑N ↑N')
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-30")] alias sup_coe_toSubmodule := sup_toSubmodule


@[simp]
theorem sSup_toSubmodule (S : Set (LieSubmodule R L M)) :
    (↑(sSup S) : Submodule R M) = sSup {(s : Submodule R M) | s ∈ S} :=
  rfl


@[deprecated (since := "2024-12-30")] alias sSup_coe_toSubmodule := sSup_toSubmodule


theorem sSup_toSubmodule_eq_iSup (S : Set (LieSubmodule R L M)) :
    (↑(sSup S) : Submodule R M) = ⨆ N ∈ S, (N : Submodule R M) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    S : Set (LieSubmodule R L M)
    ⊢ Eq (↑(SupSet.sSup S)) (iSup fun N => iSup fun h => ↑N)
  -/
  rw [sSup_toSubmodule, ← Set.image, sSup_image]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-30")] alias sSup_coe_toSubmodule' := sSup_toSubmodule_eq_iSup


@[simp]
theorem iSup_toSubmodule {ι} (p : ι → LieSubmodule R L M) :
    (↑(⨆ i, p i) : Submodule R M) = ⨆ i, (p i : Submodule R M) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    ι : Sort u_1
    p : ι → LieSubmodule R L M
    ⊢ Eq (↑(iSup fun i => p i)) (iSup fun i => ↑(p i))
  -/
  rw [iSup, sSup_toSubmodule]; ext; simp [Submodule.mem_sSup, Submodule.mem_iSup]
                                    /-
                                      🎉 no goals
                                    -/


@[deprecated (since := "2024-12-30")] alias iSup_coe_toSubmodule := iSup_toSubmodule


/-- The set of Lie submodules of a Lie module form a complete lattice. -/
instance : CompleteLattice (LieSubmodule R L M) :=
  { toSubmodule_injective.completeLattice toSubmodule sup_toSubmodule inf_toSubmodule
      sSup_toSubmodule_eq_iSup sInf_toSubmodule_eq_iInf rfl rfl with
    toPartialOrder := SetLike.instPartialOrder }


theorem mem_iSup_of_mem {ι} {b : M} {N : ι → LieSubmodule R L M} (i : ι) (h : b ∈ N i) :
    b ∈ ⨆ i, N i :=
  (le_iSup N i) h


lemma iSup_induction {ι} (N : ι → LieSubmodule R L M) {C : M → Prop} {x : M}
    (hx : x ∈ ⨆ i, N i) (hN : ∀ i, ∀ y ∈ N i, C y) (h0 : C 0)
    (hadd : ∀ y z, C y → C z → C (y + z)) : C x := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    ι : Sort u_1
    N : ι → LieSubmodule R L M
    C : M → Prop
    x : M
    hx : Membership.mem (iSup fun i => N i) x
    hN : ∀ (i : ι) (y : M), Membership.mem (N i) y → C y
    h0 : C 0
    hadd : ∀ (y z : M), C y → C z → C (HAdd.hAdd y z)
    ⊢ C x
  -/
  rw [← LieSubmodule.mem_toSubmodule, LieSubmodule.iSup_toSubmodule] at hx
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    ι : Sort u_1
    N : ι → LieSubmodule R L M
    C : M → Prop
    x : M
    hx : Membership.mem (iSup fun i => ↑(N i)) x
    hN : ∀ (i : ι) (y : M), Membership.mem (N i) y → C y
    h0 : C 0
    hadd : ∀ (y z : M), C y → C z → C (HAdd.hAdd y z)
    ⊢ C x
  -/
  exact Submodule.iSup_induction (C := C) (fun i ↦ (N i : Submodule R M)) hx hN h0 hadd
  /-
    🎉 no goals
  -/


@[elab_as_elim]
theorem iSup_induction' {ι} (N : ι → LieSubmodule R L M) {C : (x : M) → (x ∈ ⨆ i, N i) → Prop}
    (hN : ∀ (i) (x) (hx : x ∈ N i), C x (mem_iSup_of_mem i hx)) (h0 : C 0 (zero_mem _))
    (hadd : ∀ x y hx hy, C x hx → C y hy → C (x + y) (add_mem ‹_› ‹_›)) {x : M}
    (hx : x ∈ ⨆ i, N i) : C x hx := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    ι : Sort u_1
    N : ι → LieSubmodule R L M
    C : (x : M) → Membership.mem (iSup fun i => N i) x → Prop
    hN : ∀ (i : ι) (x : M) (hx : Membership.mem (N i) x), C x ⋯
    h0 : C 0 ⋯
    hadd : ∀ (x y : M) (hx : Membership.mem (iSup fun i => N i) x) (hy : Membershi …
    x : M
    hx : Membership.mem (iSup fun i => N i) x
    ⊢ C x hx
  -/
  refine Exists.elim ?_ fun (hx : x ∈ ⨆ i, N i) (hc : C x hx) => hc
  refine iSup_induction N (C := fun x : M ↦ ∃ (hx : x ∈ ⨆ i, N i), C x hx) hx
    (fun i x hx => ?_) ?_ fun x y => ?_
    /-
      case refine_1
      R : Type u
      L : Type v
      M : Type w
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      ι : Sort u_1
      N : ι → LieSubmodule R L M
      C : (x : M) → Membership.mem (iSup fun i => N i) x → Prop
      hN : ∀ (i : ι) (x : M) (hx : Membership.mem (N i) x), C x ⋯
      h0 : C 0 ⋯
      hadd : ∀ (x y : M) (hx : Membership.mem (iSup fun i => N i) x) (hy : Membershi …
      x✝ : M
      hx✝ : Membership.mem (iSup fun i => N i) x✝
      i : ι
      x : M
      hx : Membership.mem (N i) x
      ⊢ (fun x => Exists fun hx => C x hx) x
    -/
  · exact ⟨_, hN _ _ hx⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      L : Type v
      M : Type w
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      ι : Sort u_1
      N : ι → LieSubmodule R L M
      C : (x : M) → Membership.mem (iSup fun i => N i) x → Prop
      hN : ∀ (i : ι) (x : M) (hx : Membership.mem (N i) x), C x ⋯
      h0 : C 0 ⋯
      hadd : ∀ (x y : M) (hx : Membership.mem (iSup fun i => N i) x) (hy : Membershi …
      x : M
      hx : Membership.mem (iSup fun i => N i) x
      ⊢ (fun x => Exists fun hx => C x hx) 0
    -/
  · exact ⟨_, h0⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u
      L : Type v
      M : Type w
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      ι : Sort u_1
      N : ι → LieSubmodule R L M
      C : (x : M) → Membership.mem (iSup fun i => N i) x → Prop
      hN : ∀ (i : ι) (x : M) (hx : Membership.mem (N i) x), C x ⋯
      h0 : C 0 ⋯
      hadd : ∀ (x y : M) (hx : Membership.mem (iSup fun i => N i) x) (hy : Membershi …
      x✝ : M
      hx : Membership.mem (iSup fun i => N i) x✝
      x y : M
      ⊢ (fun x => Exists fun hx => C x hx) x → (fun x => Exists fun hx => C x hx) y  …
    -/
  · rintro ⟨_, Cx⟩ ⟨_, Cy⟩
    /-
      case refine_3.intro.intro
      R : Type u
      L : Type v
      M : Type w
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      ι : Sort u_1
      N : ι → LieSubmodule R L M
      C : (x : M) → Membership.mem (iSup fun i => N i) x → Prop
      hN : ∀ (i : ι) (x : M) (hx : Membership.mem (N i) x), C x ⋯
      h0 : C 0 ⋯
      hadd : ∀ (x y : M) (hx : Membership.mem (iSup fun i => N i) x) (hy : Membershi …
      x✝ : M
      hx : Membership.mem (iSup fun i => N i) x✝
      x y : M
      w✝¹ : Membership.mem (iSup fun i => N i) x
      Cx : C x w✝¹
      w✝ : Membership.mem (iSup fun i => N i) y
      Cy : C y w✝
      ⊢ Exists fun hx => C (HAdd.hAdd x y) hx
    -/
    exact ⟨_, hadd _ _ _ _ Cx Cy⟩
    /-
      🎉 no goals
    -/

-- TODO(Yaël): turn around

theorem disjoint_iff_toSubmodule :
    Disjoint N N' ↔ Disjoint (N : Submodule R M) (N' : Submodule R M) := by
  rw [disjoint_iff, disjoint_iff, ← toSubmodule_inj, inf_toSubmodule, bot_toSubmodule,
    ← disjoint_iff]


@[deprecated (since := "2024-12-30")] alias disjoint_iff_coe_toSubmodule := disjoint_iff_toSubmodule


theorem codisjoint_iff_toSubmodule :
    Codisjoint N N' ↔ Codisjoint (N : Submodule R M) (N' : Submodule R M) := by
  rw [codisjoint_iff, codisjoint_iff, ← toSubmodule_inj, sup_toSubmodule,
    top_toSubmodule, ← codisjoint_iff]


@[deprecated (since := "2024-12-30")]
alias codisjoint_iff_coe_toSubmodule := codisjoint_iff_toSubmodule


theorem isCompl_iff_toSubmodule :
    IsCompl N N' ↔ IsCompl (N : Submodule R M) (N' : Submodule R M) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N N' : LieSubmodule R L M
    ⊢ Iff (IsCompl N N') (IsCompl ↑N ↑N')
  -/
  simp only [isCompl_iff, disjoint_iff_toSubmodule, codisjoint_iff_toSubmodule]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-30")] alias isCompl_iff_coe_toSubmodule := isCompl_iff_toSubmodule


theorem iSupIndep_iff_toSubmodule {ι : Type*} {N : ι → LieSubmodule R L M} :
    iSupIndep N ↔ iSupIndep fun i ↦ (N i : Submodule R M) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    ι : Type u_1
    N : ι → LieSubmodule R L M
    ⊢ Iff (iSupIndep N) (iSupIndep fun i => ↑(N i))
  -/
  simp [iSupIndep_def, disjoint_iff_toSubmodule]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-30")]
alias iSupIndep_iff_coe_toSubmodule := iSupIndep_iff_toSubmodule


@[deprecated (since := "2024-11-24")]
alias independent_iff_toSubmodule := iSupIndep_iff_toSubmodule


@[deprecated (since := "2024-12-30")]
alias independent_iff_coe_toSubmodule := independent_iff_toSubmodule


theorem iSup_eq_top_iff_toSubmodule {ι : Sort*} {N : ι → LieSubmodule R L M} :
    ⨆ i, N i = ⊤ ↔ ⨆ i, (N i : Submodule R M) = ⊤ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    ι : Sort u_1
    N : ι → LieSubmodule R L M
    ⊢ Iff (Eq (iSup fun i => N i) Top.top) (Eq (iSup fun i => ↑(N i)) Top.top)
  -/
  rw [← iSup_toSubmodule, ← top_toSubmodule (L := L), toSubmodule_inj]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-30")]
alias iSup_eq_top_iff_coe_toSubmodule := iSup_eq_top_iff_toSubmodule


instance : Add (LieSubmodule R L M) where add := max


instance : Zero (LieSubmodule R L M) where zero := ⊥


instance : AddCommMonoid (LieSubmodule R L M) where
  add_assoc := sup_assoc
  zero_add := bot_sup_eq
  add_zero := sup_bot_eq
  add_comm := sup_comm
  nsmul := nsmulRec


@[simp]
theorem add_eq_sup : N + N' = N ⊔ N' :=
  rfl


@[simp]
theorem mem_inf (x : M) : x ∈ N ⊓ N' ↔ x ∈ N ∧ x ∈ N' := by
  rw [← mem_toSubmodule, ← mem_toSubmodule, ← mem_toSubmodule, inf_toSubmodule,
    Submodule.mem_inf]


theorem mem_sup (x : M) : x ∈ N ⊔ N' ↔ ∃ y ∈ N, ∃ z ∈ N', y + z = x := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N N' : LieSubmodule R L M
    x : M
    ⊢ Iff (Membership.mem (Max.max N N') x) (Exists fun y => And (Membership.mem N …
  -/
  rw [← mem_toSubmodule, sup_toSubmodule, Submodule.mem_sup]; exact Iff.rfl
                                                              /-
                                                                🎉 no goals
                                                              -/


                                                                 /-
                                                                   R : Type u
                                                                   L : Type v
                                                                   M : Type w
                                                                   inst✝⁴ : CommRing R
                                                                   inst✝³ : LieRing L
                                                                   inst✝² : AddCommGroup M
                                                                   inst✝¹ : Module R M
                                                                   inst✝ : LieRingModule L M
                                                                   N : LieSubmodule R L M
                                                                   ⊢ Iff (Eq N Bot.bot) (∀ (m : M), Membership.mem N m → Eq m 0)
                                                                 -/
nonrec theorem eq_bot_iff : N = ⊥ ↔ ∀ m : M, m ∈ N → m = 0 := by rw [eq_bot_iff]; exact Iff.rfl
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


instance subsingleton_of_bot : Subsingleton (LieSubmodule R L (⊥ : LieSubmodule R L M)) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N N' : LieSubmodule R L M
    ⊢ Subsingleton (LieSubmodule R L (Subtype fun x => Membership.mem Bot.bot x))
  -/
  apply subsingleton_of_bot_eq_top
  /-
    case hα
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N N' : LieSubmodule R L M
    ⊢ Eq Bot.bot Top.top
  -/
  ext ⟨_, hx⟩
  /-
    case hα.h.mk
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N N' : LieSubmodule R L M
    val✝ : M
    hx : Membership.mem Bot.bot val✝
    ⊢ Iff (Membership.mem Bot.bot ⟨val✝, hx⟩) (Membership.mem Top.top ⟨val✝, hx⟩)
  -/
  simp only [mem_bot, mk_eq_zero, mem_top, iff_true]
  /-
    case hα.h.mk
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N N' : LieSubmodule R L M
    val✝ : M
    hx : Membership.mem Bot.bot val✝
    ⊢ Eq val✝ 0
  -/
  exact hx
  /-
    🎉 no goals
  -/


instance : IsModularLattice (LieSubmodule R L M) where
  sup_inf_le_assoc_of_le _ _ := by
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      N N' x✝² x✝¹ x✝ : LieSubmodule R L M
      ⊢ LE.le x✝² x✝ → LE.le (Min.min (Max.max x✝² x✝¹) x✝) (Max.max x✝² (Min.min x✝ …
    -/
    simp only [← toSubmodule_le_toSubmodule, sup_toSubmodule, inf_toSubmodule]
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      N N' x✝² x✝¹ x✝ : LieSubmodule R L M
      ⊢ LE.le ↑x✝² ↑x✝ → LE.le (Min.min (Max.max ↑x✝² ↑x✝¹) ↑x✝) (Max.max (↑x✝²) (Mi …
    -/
    exact IsModularLattice.sup_inf_le_assoc_of_le _
    /-
      🎉 no goals
    -/


/-- The natural functor that forgets the action of `L` as an order embedding. -/
@[simps] def toSubmodule_orderEmbedding : LieSubmodule R L M ↪o Submodule R M :=
  { toFun := (↑)
    inj' := toSubmodule_injective
    map_rel_iff' := Iff.rfl }


instance wellFoundedGT_of_noetherian [IsNoetherian R M] : WellFoundedGT (LieSubmodule R L M) :=
  RelHomClass.isWellFounded (toSubmodule_orderEmbedding R L M).dual.ltEmbedding


theorem wellFoundedLT_of_isArtinian [IsArtinian R M] : WellFoundedLT (LieSubmodule R L M) :=
  RelHomClass.isWellFounded (toSubmodule_orderEmbedding R L M).ltEmbedding


instance [IsArtinian R M] : IsAtomic (LieSubmodule R L M) :=
  isAtomic_of_orderBot_wellFounded_lt <| (wellFoundedLT_of_isArtinian R L M).wf


@[simp]
theorem subsingleton_iff : Subsingleton (LieSubmodule R L M) ↔ Subsingleton M :=
  have h : Subsingleton (LieSubmodule R L M) ↔ Subsingleton (Submodule R M) := by
    rw [← subsingleton_iff_bot_eq_top, ← subsingleton_iff_bot_eq_top, ← toSubmodule_inj,
      top_toSubmodule, bot_toSubmodule]
  h.trans <| Submodule.subsingleton_iff R


@[simp]
theorem nontrivial_iff : Nontrivial (LieSubmodule R L M) ↔ Nontrivial M :=
  not_iff_not.mp
    ((not_nontrivial_iff_subsingleton.trans <| subsingleton_iff R L M).trans
      not_nontrivial_iff_subsingleton.symm)


instance [Nontrivial M] : Nontrivial (LieSubmodule R L M) :=
  (nontrivial_iff R L M).mpr ‹_›


theorem nontrivial_iff_ne_bot {N : LieSubmodule R L M} : Nontrivial N ↔ N ≠ ⊥ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N : LieSubmodule R L M
    ⊢ Iff (Nontrivial (Subtype fun x => Membership.mem N x)) (Ne N Bot.bot)
  -/
  constructor <;> contrapose!
  · rintro rfl
      ⟨⟨m₁, h₁ : m₁ ∈ (⊥ : LieSubmodule R L M)⟩, ⟨m₂, h₂ : m₂ ∈ (⊥ : LieSubmodule R L M)⟩, h₁₂⟩
    /-
      case mp.mk.intro.mk.intro.mk
      R : Type u
      L : Type v
      M : Type w
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      m₁ : M
      h₁ : Membership.mem Bot.bot m₁
      m₂ : M
      h₂ : Membership.mem Bot.bot m₂
      h₁₂ : Ne ⟨m₁, h₁⟩ ⟨m₂, h₂⟩
      ⊢ False
    -/
    simp [(LieSubmodule.mem_bot _).mp h₁, (LieSubmodule.mem_bot _).mp h₂] at h₁₂
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      L : Type v
      M : Type w
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      N : LieSubmodule R L M
      ⊢ Not (Nontrivial (Subtype fun x => Membership.mem N x)) → Eq N Bot.bot
    -/
  · rw [not_nontrivial_iff_subsingleton, LieSubmodule.eq_bot_iff]
    /-
      case mpr
      R : Type u
      L : Type v
      M : Type w
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      N : LieSubmodule R L M
      ⊢ Subsingleton (Subtype fun x => Membership.mem N x) → ∀ (m : M), Membership.m …
    -/
    rintro ⟨h⟩ m hm
    /-
      case mpr.intro
      R : Type u
      L : Type v
      M : Type w
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      N : LieSubmodule R L M
      h : ∀ (a b : Subtype fun x => Membership.mem N x), Eq a b
      m : M
      hm : Membership.mem N m
      ⊢ Eq m 0
    -/
    simpa using h ⟨m, hm⟩ ⟨_, N.zero_mem⟩
    /-
      🎉 no goals
    -/


/-- The inclusion of a Lie submodule into its ambient space is a morphism of Lie modules. -/
def incl : N →ₗ⁅R,L⁆ M :=
  { Submodule.subtype (N : Submodule R M) with map_lie' := fun {_ _} ↦ rfl }


@[simp]
theorem incl_coe : (N.incl : N →ₗ[R] M) = (N : Submodule R M).subtype :=
  rfl


@[simp]
theorem incl_apply (m : N) : N.incl m = m :=
  rfl


theorem incl_eq_val : (N.incl : N → M) = Subtype.val :=
  rfl


theorem injective_incl : Function.Injective N.incl := Subtype.coe_injective


/-- Given two nested Lie submodules `N ⊆ N'`,
the inclusion `N ↪ N'` is a morphism of Lie modules. -/
def inclusion : N →ₗ⁅R,L⁆ N' where
  __ := Submodule.inclusion (show N.toSubmodule ≤ N'.toSubmodule from h)
  map_lie' := rfl


@[simp]
theorem coe_inclusion (m : N) : (inclusion h m : M) = m :=
  rfl


theorem inclusion_apply (m : N) : inclusion h m = ⟨m.1, h m.2⟩ :=
  rfl


theorem inclusion_injective : Function.Injective (inclusion h) := fun x y ↦ by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N N' : LieSubmodule R L M
    h : LE.le N N'
    x y : Subtype fun x => Membership.mem N x
    ⊢ Eq ((LieSubmodule.inclusion h) x) ((LieSubmodule.inclusion h) y) → Eq x y
  -/
  simp only [inclusion_apply, imp_self, Subtype.mk_eq_mk, SetLike.coe_eq_coe]
  /-
    🎉 no goals
  -/


/-- The `lieSpan` of a set `s ⊆ M` is the smallest Lie submodule of `M` that contains `s`. -/
def lieSpan : LieSubmodule R L M :=
  sInf { N | s ⊆ N }


theorem mem_lieSpan {x : M} : x ∈ lieSpan R L s ↔ ∀ N : LieSubmodule R L M, s ⊆ N → x ∈ N := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    s : Set M
    x : M
    ⊢ Iff (Membership.mem (LieSubmodule.lieSpan R L s) x) (∀ (N : LieSubmodule R L …
  -/
  change x ∈ (lieSpan R L s : Set M) ↔ _; erw [sInf_coe]; exact mem_iInter₂
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem subset_lieSpan : s ⊆ lieSpan R L s := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    s : Set M
    ⊢ HasSubset.Subset s ↑(LieSubmodule.lieSpan R L s)
  -/
  intro m hm
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    s : Set M
    m : M
    hm : Membership.mem s m
    ⊢ Membership.mem (↑(LieSubmodule.lieSpan R L s)) m
  -/
  erw [mem_lieSpan]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    s : Set M
    m : M
    hm : Membership.mem s m
    ⊢ ∀ (N : LieSubmodule R L M), HasSubset.Subset s ↑N → Membership.mem N m
  -/
  intro N hN
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    s : Set M
    m : M
    hm : Membership.mem s m
    N : LieSubmodule R L M
    hN : HasSubset.Subset s ↑N
    ⊢ Membership.mem N m
  -/
  exact hN hm
  /-
    🎉 no goals
  -/


theorem submodule_span_le_lieSpan : Submodule.span R s ≤ lieSpan R L s := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    s : Set M
    ⊢ LE.le (Submodule.span R s) ↑(LieSubmodule.lieSpan R L s)
  -/
  rw [Submodule.span_le]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    s : Set M
    ⊢ HasSubset.Subset s ↑↑(LieSubmodule.lieSpan R L s)
  -/
  apply subset_lieSpan
  /-
    🎉 no goals
  -/


@[simp]
theorem lieSpan_le {N} : lieSpan R L s ≤ N ↔ s ⊆ N := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    s : Set M
    N : LieSubmodule R L M
    ⊢ Iff (LE.le (LieSubmodule.lieSpan R L s) N) (HasSubset.Subset s ↑N)
  -/
  constructor
    /-
      case mp
      R : Type u
      L : Type v
      M : Type w
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      s : Set M
      N : LieSubmodule R L M
      ⊢ LE.le (LieSubmodule.lieSpan R L s) N → HasSubset.Subset s ↑N
    -/
  · exact Subset.trans subset_lieSpan
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      L : Type v
      M : Type w
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      s : Set M
      N : LieSubmodule R L M
      ⊢ HasSubset.Subset s ↑N → LE.le (LieSubmodule.lieSpan R L s) N
    -/
  · intro hs m hm; rw [mem_lieSpan] at hm; exact hm _ hs
                                           /-
                                             🎉 no goals
                                           -/


theorem lieSpan_mono {t : Set M} (h : s ⊆ t) : lieSpan R L s ≤ lieSpan R L t := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    s t : Set M
    h : HasSubset.Subset s t
    ⊢ LE.le (LieSubmodule.lieSpan R L s) (LieSubmodule.lieSpan R L t)
  -/
  rw [lieSpan_le]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    s t : Set M
    h : HasSubset.Subset s t
    ⊢ HasSubset.Subset s ↑(LieSubmodule.lieSpan R L t)
  -/
  exact Subset.trans h subset_lieSpan
  /-
    🎉 no goals
  -/


theorem lieSpan_eq : lieSpan R L (N : Set M) = N :=
  le_antisymm (lieSpan_le.mpr rfl.subset) subset_lieSpan


theorem coe_lieSpan_submodule_eq_iff {p : Submodule R M} :
    (lieSpan R L (p : Set M) : Submodule R M) = p ↔ ∃ N : LieSubmodule R L M, ↑N = p := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    p : Submodule R M
    ⊢ Iff (Eq (↑(LieSubmodule.lieSpan R L ↑p)) p) (Exists fun N => Eq (↑N) p)
  -/
  rw [p.exists_lieSubmodule_coe_eq_iff L]; constructor <;> intro h
    /-
      case mp
      R : Type u
      L : Type v
      M : Type w
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      p : Submodule R M
      h : Eq (↑(LieSubmodule.lieSpan R L ↑p)) p
      ⊢ ∀ (x : L) (m : M), Membership.mem p m → Membership.mem p (Bracket.bracket x m)
    -/
  · intro x m hm; rw [← h, mem_toSubmodule]; exact lie_mem _ (subset_lieSpan hm)
                                             /-
                                               🎉 no goals
                                             -/
    /-
      case mpr
      R : Type u
      L : Type v
      M : Type w
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : LieRingModule L M
      p : Submodule R M
      h : ∀ (x : L) (m : M), Membership.mem p m → Membership.mem p (Bracket.bracket  …
      ⊢ Eq (↑(LieSubmodule.lieSpan R L ↑p)) p
    -/
  · rw [← toSubmodule_mk p @h, coe_toSubmodule, toSubmodule_inj, lieSpan_eq]
    /-
      🎉 no goals
    -/


/-- `lieSpan` forms a Galois insertion with the coercion from `LieSubmodule` to `Set`. -/
protected def gi : GaloisInsertion (lieSpan R L : Set M → LieSubmodule R L M) (↑) where
  choice s _ := lieSpan R L s
  gc _ _ := lieSpan_le
  le_l_u _ := subset_lieSpan
  choice_eq _ _ := rfl


@[simp]
theorem span_empty : lieSpan R L (∅ : Set M) = ⊥ :=
  (LieSubmodule.gi R L M).gc.l_bot


@[simp]
theorem span_univ : lieSpan R L (Set.univ : Set M) = ⊤ :=
  eq_top_iff.2 <| SetLike.le_def.2 <| subset_lieSpan


theorem lieSpan_eq_bot_iff : lieSpan R L s = ⊥ ↔ ∀ m ∈ s, m = (0 : M) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    s : Set M
    ⊢ Iff (Eq (LieSubmodule.lieSpan R L s) Bot.bot) (∀ (m : M), Membership.mem s m …
  -/
  rw [_root_.eq_bot_iff, lieSpan_le, bot_coe, subset_singleton_iff]
  /-
    🎉 no goals
  -/


theorem span_union (s t : Set M) : lieSpan R L (s ∪ t) = lieSpan R L s ⊔ lieSpan R L t :=
  (LieSubmodule.gi R L M).gc.l_sup


theorem span_iUnion {ι} (s : ι → Set M) : lieSpan R L (⋃ i, s i) = ⨆ i, lieSpan R L (s i) :=
  (LieSubmodule.gi R L M).gc.l_iSup


lemma isCompactElement_lieSpan_singleton (m : M) :
    CompleteLattice.IsCompactElement (lieSpan R L {m}) := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    m : M
    ⊢ CompleteLattice.IsCompactElement (LieSubmodule.lieSpan R L (Singleton.single …
  -/
  rw [CompleteLattice.isCompactElement_iff_le_of_directed_sSup_le]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    m : M
    ⊢ ∀ (s : Set (LieSubmodule R L M)), s.Nonempty → DirectedOn (fun x1 x2 => LE.l …
  -/
  intro s hne hdir hsup
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    m : M
    s : Set (LieSubmodule R L M)
    hne : s.Nonempty
    hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    hsup : LE.le (LieSubmodule.lieSpan R L (Singleton.singleton m)) (SupSet.sSup s)
    ⊢ Exists fun x => And (Membership.mem s x) (LE.le (LieSubmodule.lieSpan R L (S …
  -/
  replace hsup : m ∈ (↑(sSup s) : Set M) := (SetLike.le_def.mp hsup) (subset_lieSpan rfl)
  suffices (↑(sSup s) : Set M) = ⋃ N ∈ s, ↑N by
    obtain ⟨N : LieSubmodule R L M, hN : N ∈ s, hN' : m ∈ N⟩ := by
      simp_rw [this, Set.mem_iUnion, SetLike.mem_coe, exists_prop] at hsup; assumption
    exact ⟨N, hN, by simpa⟩
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    m : M
    s : Set (LieSubmodule R L M)
    hne : s.Nonempty
    hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    hsup : Membership.mem (↑(SupSet.sSup s)) m
    ⊢ Eq (↑(SupSet.sSup s)) (Set.iUnion fun N => Set.iUnion fun h => ↑N)
  -/
  replace hne : Nonempty s := Set.nonempty_coe_sort.mpr hne
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    m : M
    s : Set (LieSubmodule R L M)
    hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    hsup : Membership.mem (↑(SupSet.sSup s)) m
    hne : Nonempty ↑s
    ⊢ Eq (↑(SupSet.sSup s)) (Set.iUnion fun N => Set.iUnion fun h => ↑N)
  -/
  have := Submodule.coe_iSup_of_directed _ hdir.directed_val
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    m : M
    s : Set (LieSubmodule R L M)
    hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    hsup : Membership.mem (↑(SupSet.sSup s)) m
    hne : Nonempty ↑s
    this : Eq (↑(iSup fun x => ↑↑x)) (Set.iUnion fun i => ↑↑↑i)
    ⊢ Eq (↑(SupSet.sSup s)) (Set.iUnion fun N => Set.iUnion fun h => ↑N)
  -/
  simp_rw [← iSup_toSubmodule, Set.iUnion_coe_set, coe_toSubmodule] at this
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    m : M
    s : Set (LieSubmodule R L M)
    hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    hsup : Membership.mem (↑(SupSet.sSup s)) m
    hne : Nonempty ↑s
    this : Eq (↑(iSup fun i => ↑i)) (Set.iUnion fun i => Set.iUnion fun x => ↑i)
    ⊢ Eq (↑(SupSet.sSup s)) (Set.iUnion fun N => Set.iUnion fun h => ↑N)
  -/
  rw [← this, SetLike.coe_set_eq, sSup_eq_iSup, iSup_subtype]
  /-
    🎉 no goals
  -/


@[simp]
lemma sSup_image_lieSpan_singleton : sSup ((fun x ↦ lieSpan R L {x}) '' N) = N := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N : LieSubmodule R L M
    ⊢ Eq (SupSet.sSup (Set.image (fun x => LieSubmodule.lieSpan R L (Singleton.sin …
  -/
  refine le_antisymm (sSup_le <| by simp) ?_
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N : LieSubmodule R L M
    ⊢ LE.le N (SupSet.sSup (Set.image (fun x => LieSubmodule.lieSpan R L (Singleto …
  -/
  simp_rw [← toSubmodule_le_toSubmodule, sSup_toSubmodule, Set.mem_image, SetLike.mem_coe]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N : LieSubmodule R L M
    ⊢ LE.le (↑N) (SupSet.sSup (setOf fun x => Exists fun s => And (Exists fun x => …
  -/
  refine fun m hm ↦ Submodule.mem_sSup.mpr fun N' hN' ↦ ?_
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N : LieSubmodule R L M
    m : M
    hm : Membership.mem (↑N) m
    N' : Submodule R M
    hN' : ∀ (p : Submodule R M), Membership.mem (setOf fun x => Exists fun s => An …
    ⊢ Membership.mem N' m
  -/
  replace hN' : ∀ m ∈ N, lieSpan R L {m} ≤ N' := by simpa using hN'
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N : LieSubmodule R L M
    m : M
    hm : Membership.mem (↑N) m
    N' : Submodule R M
    hN' : ∀ (m : M), Membership.mem N m → LE.le (↑(LieSubmodule.lieSpan R L (Singl …
    ⊢ Membership.mem N' m
  -/
  exact hN' _ hm (subset_lieSpan rfl)
  /-
    🎉 no goals
  -/


instance instIsCompactlyGenerated : IsCompactlyGenerated (LieSubmodule R L M) :=
  ⟨fun N ↦ ⟨(fun x ↦ lieSpan R L {x}) '' N, fun _ ⟨m, _, hm⟩ ↦
    hm ▸ isCompactElement_lieSpan_singleton R L m, N.sSup_image_lieSpan_singleton⟩⟩


/-- A morphism of Lie modules `f : M → M'` pushes forward Lie submodules of `M` to Lie submodules
of `M'`. -/
def map : LieSubmodule R L M' :=
  { (N : Submodule R M).map (f : M →ₗ[R] M') with
    lie_mem := fun {x m'} h ↦ by
      /-
        R : Type u
        L : Type v
        L' : Type w₂
        M : Type w
        M' : Type w₁
        inst✝⁹ : CommRing R
        inst✝⁸ : LieRing L
        inst✝⁷ : LieRing L'
        inst✝⁶ : LieAlgebra R L'
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : LieRingModule L M
        inst✝² : AddCommGroup M'
        inst✝¹ : Module R M'
        inst✝ : LieRingModule L M'
        f : LieModuleHom R L M M'
        N N₂ : LieSubmodule R L M
        N' : LieSubmodule R L M'
        x : L
        m' : M'
        h : Membership.mem __src✝.carrier m'
        ⊢ Membership.mem __src✝.carrier (Bracket.bracket x m')
      -/
      rcases h with ⟨m, hm, hfm⟩; use ⁅x, m⁆; constructor
        /-
          case h.left
          R : Type u
          L : Type v
          L' : Type w₂
          M : Type w
          M' : Type w₁
          inst✝⁹ : CommRing R
          inst✝⁸ : LieRing L
          inst✝⁷ : LieRing L'
          inst✝⁶ : LieAlgebra R L'
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : LieRingModule L M
          inst✝² : AddCommGroup M'
          inst✝¹ : Module R M'
          inst✝ : LieRingModule L M'
          f : LieModuleHom R L M M'
          N N₂ : LieSubmodule R L M
          N' : LieSubmodule R L M'
          x : L
          m' : M'
          m : M
          hm : Membership.mem (↑↑N) m
          hfm : Eq (↑f m) m'
          ⊢ Membership.mem (↑↑N) (Bracket.bracket x m)
        -/
      · apply N.lie_mem hm
        /-
          🎉 no goals
        -/
        /-
          case h.right
          R : Type u
          L : Type v
          L' : Type w₂
          M : Type w
          M' : Type w₁
          inst✝⁹ : CommRing R
          inst✝⁸ : LieRing L
          inst✝⁷ : LieRing L'
          inst✝⁶ : LieAlgebra R L'
          inst✝⁵ : AddCommGroup M
          inst✝⁴ : Module R M
          inst✝³ : LieRingModule L M
          inst✝² : AddCommGroup M'
          inst✝¹ : Module R M'
          inst✝ : LieRingModule L M'
          f : LieModuleHom R L M M'
          N N₂ : LieSubmodule R L M
          N' : LieSubmodule R L M'
          x : L
          m' : M'
          m : M
          hm : Membership.mem (↑↑N) m
          hfm : Eq (↑f m) m'
          ⊢ Eq (↑f (Bracket.bracket x m)) (Bracket.bracket x m')
        -/
      · norm_cast at hfm; simp [hfm] }
                          /-
                            🎉 no goals
                          -/


@[simp] theorem coe_map : (N.map f : Set M') = f '' N := rfl


@[simp]
theorem toSubmodule_map : (N.map f : Submodule R M') = (N : Submodule R M).map (f : M →ₗ[R] M') :=
  rfl


@[deprecated (since := "2024-12-30")] alias coeSubmodule_map := toSubmodule_map


/-- A morphism of Lie modules `f : M → M'` pulls back Lie submodules of `M'` to Lie submodules of
`M`. -/
def comap : LieSubmodule R L M :=
  { (N' : Submodule R M').comap (f : M →ₗ[R] M') with
    lie_mem := fun {x m} h ↦ by
      /-
        R : Type u
        L : Type v
        L' : Type w₂
        M : Type w
        M' : Type w₁
        inst✝⁹ : CommRing R
        inst✝⁸ : LieRing L
        inst✝⁷ : LieRing L'
        inst✝⁶ : LieAlgebra R L'
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : LieRingModule L M
        inst✝² : AddCommGroup M'
        inst✝¹ : Module R M'
        inst✝ : LieRingModule L M'
        f : LieModuleHom R L M M'
        N N₂ : LieSubmodule R L M
        N' : LieSubmodule R L M'
        x : L
        m : M
        h : Membership.mem __src✝.carrier m
        ⊢ Membership.mem __src✝.carrier (Bracket.bracket x m)
      -/
      suffices ⁅x, f m⁆ ∈ N' by simp [this]
      /-
        R : Type u
        L : Type v
        L' : Type w₂
        M : Type w
        M' : Type w₁
        inst✝⁹ : CommRing R
        inst✝⁸ : LieRing L
        inst✝⁷ : LieRing L'
        inst✝⁶ : LieAlgebra R L'
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : Module R M
        inst✝³ : LieRingModule L M
        inst✝² : AddCommGroup M'
        inst✝¹ : Module R M'
        inst✝ : LieRingModule L M'
        f : LieModuleHom R L M M'
        N N₂ : LieSubmodule R L M
        N' : LieSubmodule R L M'
        x : L
        m : M
        h : Membership.mem __src✝.carrier m
        ⊢ Membership.mem N' (Bracket.bracket x (f m))
      -/
      apply N'.lie_mem h }
      /-
        🎉 no goals
      -/


@[simp]
theorem toSubmodule_comap :
    (N'.comap f : Submodule R M) = (N' : Submodule R M').comap (f : M →ₗ[R] M') :=
  rfl


@[deprecated (since := "2024-12-30")] alias coeSubmodule_comap := toSubmodule_comap


theorem map_le_iff_le_comap : map f N ≤ N' ↔ N ≤ comap f N' :=
  Set.image_subset_iff


theorem gc_map_comap : GaloisConnection (map f) (comap f) := fun _ _ ↦ map_le_iff_le_comap


theorem map_inf_le : (N ⊓ N₂).map f ≤ N.map f ⊓ N₂.map f :=
  Set.image_inter_subset f N N₂


theorem map_inf (hf : Function.Injective f) :
    (N ⊓ N₂).map f = N.map f ⊓ N₂.map f :=
  SetLike.coe_injective <| Set.image_inter hf


@[simp]
theorem map_sup : (N ⊔ N₂).map f = N.map f ⊔ N₂.map f :=
  (gc_map_comap f).l_sup


@[simp]
theorem comap_inf {N₂' : LieSubmodule R L M'} :
    (N' ⊓ N₂').comap f = N'.comap f ⊓ N₂'.comap f :=
  rfl


@[simp]
theorem map_iSup {ι : Sort*} (N : ι → LieSubmodule R L M) :
    (⨆ i, N i).map f = ⨆ i, (N i).map f :=
  (gc_map_comap f : GaloisConnection (map f) (comap f)).l_iSup


@[simp]
theorem mem_map (m' : M') : m' ∈ N.map f ↔ ∃ m, m ∈ N ∧ f m = m' :=
  Submodule.mem_map


theorem mem_map_of_mem {m : M} (h : m ∈ N) : f m ∈ N.map f :=
  Set.mem_image_of_mem _ h


@[simp]
theorem mem_comap {m : M} : m ∈ comap f N' ↔ f m ∈ N' :=
  Iff.rfl


theorem comap_incl_eq_top : N₂.comap N.incl = ⊤ ↔ N ≤ N₂ := by
  rw [← LieSubmodule.toSubmodule_inj, LieSubmodule.toSubmodule_comap, LieSubmodule.incl_coe,
    LieSubmodule.top_toSubmodule, Submodule.comap_subtype_eq_top, toSubmodule_le_toSubmodule]


theorem comap_incl_eq_bot : N₂.comap N.incl = ⊥ ↔ N ⊓ N₂ = ⊥ := by
  simp only [← toSubmodule_inj, toSubmodule_comap, incl_coe, bot_toSubmodule,
    inf_toSubmodule]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N N₂ : LieSubmodule R L M
    ⊢ Iff (Eq (Submodule.comap (↑N).subtype ↑N₂) Bot.bot) (Eq (Min.min ↑N ↑N₂) Bot …
  -/
  rw [← Submodule.disjoint_iff_comap_eq_bot, disjoint_iff]
  /-
    🎉 no goals
  -/


@[gcongr, mono]
theorem map_mono (h : N ≤ N₂) : N.map f ≤ N₂.map f :=
  Set.image_subset _ h


theorem map_comp
    {M'' : Type*} [AddCommGroup M''] [Module R M''] [LieRingModule L M''] {g : M' →ₗ⁅R,L⁆ M''} :
    N.map (g.comp f) = (N.map f).map g :=
  SetLike.coe_injective <| by
    /-
      R : Type u
      L : Type v
      M : Type w
      M' : Type w₁
      inst✝¹⁰ : CommRing R
      inst✝⁹ : LieRing L
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : Module R M
      inst✝⁶ : LieRingModule L M
      inst✝⁵ : AddCommGroup M'
      inst✝⁴ : Module R M'
      inst✝³ : LieRingModule L M'
      f : LieModuleHom R L M M'
      N : LieSubmodule R L M
      M'' : Type u_1
      inst✝² : AddCommGroup M''
      inst✝¹ : Module R M''
      inst✝ : LieRingModule L M''
      g : LieModuleHom R L M' M''
      ⊢ Eq ↑(LieSubmodule.map (g.comp f) N) ↑(LieSubmodule.map g (LieSubmodule.map f …
    -/
    simp only [← Set.image_comp, coe_map, LinearMap.coe_comp, LieModuleHom.coe_comp]
    /-
      🎉 no goals
    -/


@[simp]
                                                 /-
                                                   R : Type u
                                                   L : Type v
                                                   M : Type w
                                                   inst✝⁴ : CommRing R
                                                   inst✝³ : LieRing L
                                                   inst✝² : AddCommGroup M
                                                   inst✝¹ : Module R M
                                                   inst✝ : LieRingModule L M
                                                   N : LieSubmodule R L M
                                                   ⊢ Eq (LieSubmodule.map LieModuleHom.id N) N
                                                 -/
theorem map_id : N.map LieModuleHom.id = N := by ext; simp
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp] theorem map_bot :
    (⊥ : LieSubmodule R L M).map f = ⊥ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    M' : Type w₁
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : AddCommGroup M'
    inst✝¹ : Module R M'
    inst✝ : LieRingModule L M'
    f : LieModuleHom R L M M'
    ⊢ Eq (LieSubmodule.map f Bot.bot) Bot.bot
  -/
  ext m; simp [eq_comm]
         /-
           🎉 no goals
         -/


lemma map_le_map_iff (hf : Function.Injective f) :
    N.map f ≤ N₂.map f ↔ N ≤ N₂ :=
  Set.image_subset_image_iff hf


lemma map_injective_of_injective (hf : Function.Injective f) :
    Function.Injective (map f) := fun {N N'} h ↦
                                                    /-
                                                      R : Type u
                                                      L : Type v
                                                      M : Type w
                                                      M' : Type w₁
                                                      inst✝⁷ : CommRing R
                                                      inst✝⁶ : LieRing L
                                                      inst✝⁵ : AddCommGroup M
                                                      inst✝⁴ : Module R M
                                                      inst✝³ : LieRingModule L M
                                                      inst✝² : AddCommGroup M'
                                                      inst✝¹ : Module R M'
                                                      inst✝ : LieRingModule L M'
                                                      f : LieModuleHom R L M M'
                                                      hf : Function.Injective ⇑f
                                                      N N' : LieSubmodule R L M
                                                      h : Eq (LieSubmodule.map f N) (LieSubmodule.map f N')
                                                      ⊢ Eq (Set.image ⇑f ↑N) (Set.image ⇑f ↑N')
                                                    -/
  SetLike.coe_injective <| hf.image_injective <| by simp only [← coe_map, h]
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- An injective morphism of Lie modules embeds the lattice of submodules of the domain into that
of the target. -/
@[simps] def mapOrderEmbedding {f : M →ₗ⁅R,L⁆ M'} (hf : Function.Injective f) :
  LieSubmodule R L M ↪o LieSubmodule R L M' where
    toFun := LieSubmodule.map f
    inj' := map_injective_of_injective hf
    map_rel_iff' := Set.image_subset_image_iff hf


variable (N) in
/-- For an injective morphism of Lie modules, any Lie submodule is equivalent to its image. -/
noncomputable def equivMapOfInjective (hf : Function.Injective f) :
    N ≃ₗ⁅R,L⁆ N.map f :=
  { Submodule.equivMapOfInjective (f : M →ₗ[R] M') hf N with
    -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to specify `invFun` explicitly this way, otherwise we'd get a type mismatch
                 /-
                   R : Type u
                   L : Type v
                   L' : Type w₂
                   M : Type w
                   M' : Type w₁
                   inst✝⁹ : CommRing R
                   inst✝⁸ : LieRing L
                   inst✝⁷ : LieRing L'
                   inst✝⁶ : LieAlgebra R L'
                   inst✝⁵ : AddCommGroup M
                   inst✝⁴ : Module R M
                   inst✝³ : LieRingModule L M
                   inst✝² : AddCommGroup M'
                   inst✝¹ : Module R M'
                   inst✝ : LieRingModule L M'
                   f : LieModuleHom R L M M'
                   N N₂ : LieSubmodule R L M
                   N' : LieSubmodule R L M'
                   hf : Function.Injective ⇑f
                   ⊢ (Subtype fun x => Membership.mem (LieSubmodule.map f N) x) → Subtype fun x = …
                 -/
                   /-
                     R : Type u
                     L : Type v
                     L' : Type w₂
                     M : Type w
                     M' : Type w₁
                     inst✝⁹ : CommRing R
                     inst✝⁸ : LieRing L
                     inst✝⁷ : LieRing L'
                     inst✝⁶ : LieAlgebra R L'
                     inst✝⁵ : AddCommGroup M
                     inst✝⁴ : Module R M
                     inst✝³ : LieRingModule L M
                     inst✝² : AddCommGroup M'
                     inst✝¹ : Module R M'
                     inst✝ : LieRingModule L M'
                     f : LieModuleHom R L M M'
                     N N₂ : LieSubmodule R L M
                     N' : LieSubmodule R L M'
                     hf : Function.Injective ⇑f
                     ⊢ ∀ {x : L} {m : Subtype fun x => Membership.mem N x}, Eq ((↑__src✝).toFun (Br …
                   -/
    invFun := by exact DFunLike.coe (Submodule.equivMapOfInjective (f : M →ₗ[R] M') hf N).symm
                                                  /-
                                                    🎉 no goals
                                                  -/
                 /-
                   🎉 no goals
                 -/
    map_lie' := by rintro x ⟨m, hm : m ∈ N⟩; ext; exact f.map_lie x m }


/-- An equivalence of Lie modules yields an order-preserving equivalence of their lattices of Lie
Submodules. -/
@[simps] def orderIsoMapComap (e : M ≃ₗ⁅R,L⁆ M') :
    LieSubmodule R L M ≃o LieSubmodule R L M' where
  toFun := map e
  invFun := comap e
                         /-
                           R : Type u
                           L : Type v
                           L' : Type w₂
                           M : Type w
                           M' : Type w₁
                           inst✝⁹ : CommRing R
                           inst✝⁸ : LieRing L
                           inst✝⁷ : LieRing L'
                           inst✝⁶ : LieAlgebra R L'
                           inst✝⁵ : AddCommGroup M
                           inst✝⁴ : Module R M
                           inst✝³ : LieRingModule L M
                           inst✝² : AddCommGroup M'
                           inst✝¹ : Module R M'
                           inst✝ : LieRingModule L M'
                           f : LieModuleHom R L M M'
                           N✝ N₂ : LieSubmodule R L M
                           N' : LieSubmodule R L M'
                           e : LieModuleEquiv R L M M'
                           N : LieSubmodule R L M
                           ⊢ Eq (LieSubmodule.comap e.toLieModuleHom (LieSubmodule.map e.toLieModuleHom N …
                         -/
  left_inv := fun N ↦ by ext; simp
                              /-
                                🎉 no goals
                              -/
                          /-
                            R : Type u
                            L : Type v
                            L' : Type w₂
                            M : Type w
                            M' : Type w₁
                            inst✝⁹ : CommRing R
                            inst✝⁸ : LieRing L
                            inst✝⁷ : LieRing L'
                            inst✝⁶ : LieAlgebra R L'
                            inst✝⁵ : AddCommGroup M
                            inst✝⁴ : Module R M
                            inst✝³ : LieRingModule L M
                            inst✝² : AddCommGroup M'
                            inst✝¹ : Module R M'
                            inst✝ : LieRingModule L M'
                            f : LieModuleHom R L M M'
                            N✝ N₂ : LieSubmodule R L M
                            N' : LieSubmodule R L M'
                            e : LieModuleEquiv R L M M'
                            N : LieSubmodule R L M'
                            ⊢ Eq (LieSubmodule.map e.toLieModuleHom (LieSubmodule.comap e.toLieModuleHom N …
                          -/
  right_inv := fun N ↦ by ext; simp [e.apply_eq_iff_eq_symm_apply]
                               /-
                                 🎉 no goals
                               -/
  map_rel_iff' := fun {_ _} ↦ Set.image_subset_image_iff e.injective


@[simp]
theorem top_toLieSubalgebra : ((⊤ : LieIdeal R L) : LieSubalgebra R L) = ⊤ :=
  rfl


@[deprecated (since := "2024-12-30")] alias top_coe_lieSubalgebra := top_toLieSubalgebra


/-- A morphism of Lie algebras `f : L → L'` pushes forward Lie ideals of `L` to Lie ideals of `L'`.

Note that unlike `LieSubmodule.map`, we must take the `lieSpan` of the image. Mathematically
this is because although `f` makes `L'` into a Lie module over `L`, in general the `L` submodules of
`L'` are not the same as the ideals of `L'`. -/
def map : LieIdeal R L' :=
  LieSubmodule.lieSpan R L' <| (I : Submodule R L).map (f : L →ₗ[R] L')


/-- A morphism of Lie algebras `f : L → L'` pulls back Lie ideals of `L'` to Lie ideals of `L`.

Note that `f` makes `L'` into a Lie module over `L` (turning `f` into a morphism of Lie modules)
and so this is a special case of `LieSubmodule.comap` but we do not exploit this fact. -/
def comap : LieIdeal R L :=
  { (J : Submodule R L').comap (f : L →ₗ[R] L') with
    lie_mem := fun {x y} h ↦ by
      suffices ⁅f x, f y⁆ ∈ J by
        simp only [AddSubsemigroup.mem_carrier, AddSubmonoid.mem_toSubsemigroup,
          Submodule.mem_toAddSubmonoid, Submodule.mem_comap, LieHom.coe_toLinearMap, LieHom.map_lie,
          LieSubalgebra.mem_toSubmodule]
        exact this
      /-
        R : Type u
        L : Type v
        L' : Type w₂
        M : Type w
        M' : Type w₁
        inst✝¹² : CommRing R
        inst✝¹¹ : LieRing L
        inst✝¹⁰ : LieRing L'
        inst✝⁹ : LieAlgebra R L'
        inst✝⁸ : AddCommGroup M
        inst✝⁷ : Module R M
        inst✝⁶ : LieRingModule L M
        inst✝⁵ : AddCommGroup M'
        inst✝⁴ : Module R M'
        inst✝³ : LieRingModule L M'
        inst✝² : LieAlgebra R L
        inst✝¹ : LieModule R L M
        inst✝ : LieModule R L M'
        f : LieHom R L L'
        I I₂ : LieIdeal R L
        J : LieIdeal R L'
        x y : L
        h : Membership.mem __src✝.carrier y
        ⊢ Membership.mem J (Bracket.bracket (f x) (f y))
      -/
      apply J.lie_mem h }
      /-
        🎉 no goals
      -/


@[simp]
theorem map_toSubmodule (h : ↑(map f I) = f '' I) :
    LieSubmodule.toSubmodule (map f I) = (LieSubmodule.toSubmodule I).map (f : L →ₗ[R] L') := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    h : Eq (↑(LieIdeal.map f I)) (Set.image ⇑f ↑I)
    ⊢ Eq (↑(LieIdeal.map f I)) (Submodule.map ↑f ↑I)
  -/
  rw [SetLike.ext'_iff, LieSubmodule.coe_toSubmodule, h, Submodule.map_coe]; rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[deprecated (since := "2024-12-30")] alias map_coeSubmodule := map_toSubmodule


@[simp]
theorem comap_toSubmodule :
    (LieSubmodule.toSubmodule (comap f J)) = (LieSubmodule.toSubmodule J).comap (f : L →ₗ[R] L') :=
  rfl


@[deprecated (since := "2024-12-30")] alias comap_coeSubmodule := comap_toSubmodule


theorem map_le : map f I ≤ J ↔ f '' I ⊆ J :=
  LieSubmodule.lieSpan_le


theorem mem_map {x : L} (hx : x ∈ I) : f x ∈ map f I := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    x : L
    hx : Membership.mem I x
    ⊢ Membership.mem (LieIdeal.map f I) (f x)
  -/
  apply LieSubmodule.subset_lieSpan
  /-
    case a
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    x : L
    hx : Membership.mem I x
    ⊢ Membership.mem (↑(Submodule.map (↑f) (LieIdeal.toLieSubalgebra R L I).toSubm …
  -/
  use x
  /-
    case h
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    x : L
    hx : Membership.mem I x
    ⊢ And (Membership.mem (↑(LieIdeal.toLieSubalgebra R L I).toSubmodule) x) (Eq ( …
  -/
  exact ⟨hx, rfl⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_comap {x : L} : x ∈ comap f J ↔ f x ∈ J :=
  Iff.rfl


theorem map_le_iff_le_comap : map f I ≤ J ↔ I ≤ comap f J := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    J : LieIdeal R L'
    ⊢ Iff (LE.le (LieIdeal.map f I) J) (LE.le I (LieIdeal.comap f J))
  -/
  rw [map_le]
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    J : LieIdeal R L'
    ⊢ Iff (HasSubset.Subset (Set.image ⇑f ↑I) ↑J) (LE.le I (LieIdeal.comap f J))
  -/
  exact Set.image_subset_iff
  /-
    🎉 no goals
  -/


@[simp]
theorem map_sup : (I ⊔ I₂).map f = I.map f ⊔ I₂.map f :=
  (gc_map_comap f).l_sup


                                                   /-
                                                     R : Type u
                                                     L : Type v
                                                     L' : Type w₂
                                                     inst✝⁴ : CommRing R
                                                     inst✝³ : LieRing L
                                                     inst✝² : LieRing L'
                                                     inst✝¹ : LieAlgebra R L'
                                                     inst✝ : LieAlgebra R L
                                                     f : LieHom R L L'
                                                     J : LieIdeal R L'
                                                     ⊢ LE.le (LieIdeal.map f (LieIdeal.comap f J)) J
                                                   -/
theorem map_comap_le : map f (comap f J) ≤ J := by rw [map_le_iff_le_comap]
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- See also `LieIdeal.map_comap_eq`. -/
                                                   /-
                                                     R : Type u
                                                     L : Type v
                                                     L' : Type w₂
                                                     inst✝⁴ : CommRing R
                                                     inst✝³ : LieRing L
                                                     inst✝² : LieRing L'
                                                     inst✝¹ : LieAlgebra R L'
                                                     inst✝ : LieAlgebra R L
                                                     f : LieHom R L L'
                                                     I : LieIdeal R L
                                                     ⊢ LE.le I (LieIdeal.comap f (LieIdeal.map f I))
                                                   -/
theorem comap_map_le : I ≤ comap f (map f I) := by rw [← map_le_iff_le_comap]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[mono]
theorem map_mono : Monotone (map f) := fun I₁ I₂ h ↦ by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I₁ I₂ : LieIdeal R L
    h : LE.le I₁ I₂
    ⊢ LE.le (LieIdeal.map f I₁) (LieIdeal.map f I₂)
  -/
  rw [SetLike.le_def] at h
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I₁ I₂ : LieIdeal R L
    h : ∀ ⦃x : L⦄, Membership.mem I₁ x → Membership.mem I₂ x
    ⊢ LE.le (LieIdeal.map f I₁) (LieIdeal.map f I₂)
  -/
  apply LieSubmodule.lieSpan_mono (Set.image_subset (⇑f) h)
  /-
    🎉 no goals
  -/


@[mono]
theorem comap_mono : Monotone (comap f) := fun J₁ J₂ h ↦ by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    J₁ J₂ : LieIdeal R L'
    h : LE.le J₁ J₂
    ⊢ LE.le (LieIdeal.comap f J₁) (LieIdeal.comap f J₂)
  -/
  rw [← SetLike.coe_subset_coe] at h ⊢
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    J₁ J₂ : LieIdeal R L'
    h : HasSubset.Subset ↑J₁ ↑J₂
    ⊢ HasSubset.Subset ↑(LieIdeal.comap f J₁) ↑(LieIdeal.comap f J₂)
  -/
  dsimp only [SetLike.coe]
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    J₁ J₂ : LieIdeal R L'
    h : HasSubset.Subset ↑J₁ ↑J₂
    ⊢ HasSubset.Subset (↑(LieIdeal.comap f J₁)).carrier (↑(LieIdeal.comap f J₂)).c …
  -/
  exact Set.preimage_mono h
  /-
    🎉 no goals
  -/


theorem map_of_image (h : f '' I = J) : I.map f = J := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    J : LieIdeal R L'
    h : Eq (Set.image ⇑f ↑I) ↑J
    ⊢ Eq (LieIdeal.map f I) J
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      L : Type v
      L' : Type w₂
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieRing L'
      inst✝¹ : LieAlgebra R L'
      inst✝ : LieAlgebra R L
      f : LieHom R L L'
      I : LieIdeal R L
      J : LieIdeal R L'
      h : Eq (Set.image ⇑f ↑I) ↑J
      ⊢ LE.le (LieIdeal.map f I) J
    -/
  · erw [LieSubmodule.lieSpan_le, Submodule.map_coe, h]
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u
      L : Type v
      L' : Type w₂
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieRing L'
      inst✝¹ : LieAlgebra R L'
      inst✝ : LieAlgebra R L
      f : LieHom R L L'
      I : LieIdeal R L
      J : LieIdeal R L'
      h : Eq (Set.image ⇑f ↑I) ↑J
      ⊢ LE.le J (LieIdeal.map f I)
    -/
  · rw [← SetLike.coe_subset_coe, ← h]; exact LieSubmodule.subset_lieSpan
                                        /-
                                          🎉 no goals
                                        -/


/-- Note that this is not a special case of `LieSubmodule.subsingleton_of_bot`. Indeed, given
`I : LieIdeal R L`, in general the two lattices `LieIdeal R I` and `LieSubmodule R L I` are
different (though the latter does naturally inject into the former).

In other words, in general, ideals of `I`, regarded as a Lie algebra in its own right, are not the
same as ideals of `L` contained in `I`. -/
instance subsingleton_of_bot : Subsingleton (LieIdeal R (⊥ : LieIdeal R L)) := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    M : Type w
    M' : Type w₁
    inst✝¹² : CommRing R
    inst✝¹¹ : LieRing L
    inst✝¹⁰ : LieRing L'
    inst✝⁹ : LieAlgebra R L'
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : AddCommGroup M'
    inst✝⁴ : Module R M'
    inst✝³ : LieRingModule L M'
    inst✝² : LieAlgebra R L
    inst✝¹ : LieModule R L M
    inst✝ : LieModule R L M'
    f : LieHom R L L'
    I I₂ : LieIdeal R L
    J : LieIdeal R L'
    ⊢ Subsingleton (LieIdeal R (Subtype fun x => Membership.mem Bot.bot x))
  -/
  apply subsingleton_of_bot_eq_top
  /-
    case hα
    R : Type u
    L : Type v
    L' : Type w₂
    M : Type w
    M' : Type w₁
    inst✝¹² : CommRing R
    inst✝¹¹ : LieRing L
    inst✝¹⁰ : LieRing L'
    inst✝⁹ : LieAlgebra R L'
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : AddCommGroup M'
    inst✝⁴ : Module R M'
    inst✝³ : LieRingModule L M'
    inst✝² : LieAlgebra R L
    inst✝¹ : LieModule R L M
    inst✝ : LieModule R L M'
    f : LieHom R L L'
    I I₂ : LieIdeal R L
    J : LieIdeal R L'
    ⊢ Eq Bot.bot Top.top
  -/
  ext ⟨x, hx⟩
  /-
    case hα.h.mk
    R : Type u
    L : Type v
    L' : Type w₂
    M : Type w
    M' : Type w₁
    inst✝¹² : CommRing R
    inst✝¹¹ : LieRing L
    inst✝¹⁰ : LieRing L'
    inst✝⁹ : LieAlgebra R L'
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : AddCommGroup M'
    inst✝⁴ : Module R M'
    inst✝³ : LieRingModule L M'
    inst✝² : LieAlgebra R L
    inst✝¹ : LieModule R L M
    inst✝ : LieModule R L M'
    f : LieHom R L L'
    I I₂ : LieIdeal R L
    J : LieIdeal R L'
    x : L
    hx : Membership.mem Bot.bot x
    ⊢ Iff (Membership.mem Bot.bot ⟨x, hx⟩) (Membership.mem Top.top ⟨x, hx⟩)
  -/
  rw [LieSubmodule.mem_bot] at hx
  /-
    case hα.h.mk
    R : Type u
    L : Type v
    L' : Type w₂
    M : Type w
    M' : Type w₁
    inst✝¹² : CommRing R
    inst✝¹¹ : LieRing L
    inst✝¹⁰ : LieRing L'
    inst✝⁹ : LieAlgebra R L'
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : AddCommGroup M'
    inst✝⁴ : Module R M'
    inst✝³ : LieRingModule L M'
    inst✝² : LieAlgebra R L
    inst✝¹ : LieModule R L M
    inst✝ : LieModule R L M'
    f : LieHom R L L'
    I I₂ : LieIdeal R L
    J : LieIdeal R L'
    x : L
    hx✝ : Membership.mem Bot.bot x
    hx : Eq x 0
    ⊢ Iff (Membership.mem Bot.bot ⟨x, hx✝⟩) (Membership.mem Top.top ⟨x, hx✝⟩)
  -/
  subst hx
  /-
    case hα.h.mk
    R : Type u
    L : Type v
    L' : Type w₂
    M : Type w
    M' : Type w₁
    inst✝¹² : CommRing R
    inst✝¹¹ : LieRing L
    inst✝¹⁰ : LieRing L'
    inst✝⁹ : LieAlgebra R L'
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : LieRingModule L M
    inst✝⁵ : AddCommGroup M'
    inst✝⁴ : Module R M'
    inst✝³ : LieRingModule L M'
    inst✝² : LieAlgebra R L
    inst✝¹ : LieModule R L M
    inst✝ : LieModule R L M'
    f : LieHom R L L'
    I I₂ : LieIdeal R L
    J : LieIdeal R L'
    hx : Membership.mem Bot.bot 0
    ⊢ Iff (Membership.mem Bot.bot ⟨0, hx⟩) (Membership.mem Top.top ⟨0, hx⟩)
  -/
  simp only [LieSubmodule.mk_eq_zero, LieSubmodule.mem_bot, LieSubmodule.mem_top]
  /-
    🎉 no goals
  -/


/-- The kernel of a morphism of Lie algebras, as an ideal in the domain. -/
def ker : LieIdeal R L :=
  LieIdeal.comap f ⊥


/-- The range of a morphism of Lie algebras as an ideal in the codomain. -/
def idealRange : LieIdeal R L' :=
  LieSubmodule.lieSpan R L' f.range


theorem idealRange_eq_lieSpan_range : f.idealRange = LieSubmodule.lieSpan R L' f.range :=
  rfl


theorem idealRange_eq_map : f.idealRange = LieIdeal.map f ⊤ := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    ⊢ Eq f.idealRange (LieIdeal.map f Top.top)
  -/
  ext
  /-
    case h
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    m✝ : L'
    ⊢ Iff (Membership.mem f.idealRange m✝) (Membership.mem (LieIdeal.map f Top.top …
  -/
  simp only [idealRange, range_eq_map]
  /-
    case h
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    m✝ : L'
    ⊢ Iff (Membership.mem (LieSubmodule.lieSpan R L' ↑(LieSubalgebra.map f Top.top …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The condition that the range of a morphism of Lie algebras is an ideal. -/
def IsIdealMorphism : Prop :=
  (f.idealRange : LieSubalgebra R L') = f.range


@[simp]
theorem isIdealMorphism_def : f.IsIdealMorphism ↔ (f.idealRange : LieSubalgebra R L') = f.range :=
  Iff.rfl


variable {f} in
theorem IsIdealMorphism.eq (hf : f.IsIdealMorphism) : f.idealRange = f.range := hf


theorem isIdealMorphism_iff : f.IsIdealMorphism ↔ ∀ (x : L') (y : L), ∃ z : L, ⁅x, f y⁆ = f z := by
  simp only [isIdealMorphism_def, idealRange_eq_lieSpan_range, ←
    LieSubalgebra.toSubmodule_inj, ← f.range.coe_toSubmodule,
    LieIdeal.toLieSubalgebra_toSubmodule, LieSubmodule.coe_lieSpan_submodule_eq_iff,
    LieSubalgebra.mem_toSubmodule, mem_range, exists_imp,
    Submodule.exists_lieSubmodule_coe_eq_iff]
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    ⊢ Iff (∀ (x m : L') (x_1 : L), Eq (f x_1) m → Exists fun y => Eq (f y) (Bracke …
  -/
  constructor
    /-
      case mp
      R : Type u
      L : Type v
      L' : Type w₂
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieRing L'
      inst✝¹ : LieAlgebra R L'
      inst✝ : LieAlgebra R L
      f : LieHom R L L'
      ⊢ (∀ (x m : L') (x_1 : L), Eq (f x_1) m → Exists fun y => Eq (f y) (Bracket.br …
    -/
  · intro h x y; obtain ⟨z, hz⟩ := h x (f y) y rfl; use z; exact hz.symm
                                                           /-
                                                             🎉 no goals
                                                           -/
    /-
      case mpr
      R : Type u
      L : Type v
      L' : Type w₂
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieRing L'
      inst✝¹ : LieAlgebra R L'
      inst✝ : LieAlgebra R L
      f : LieHom R L L'
      ⊢ (∀ (x : L') (y : L), Exists fun z => Eq (Bracket.bracket x (f y)) (f z)) → ∀ …
    -/
  · intro h x y z hz; obtain ⟨w, hw⟩ := h x z; use w; rw [← hw, hz]
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem range_subset_idealRange : (f.range : Set L') ⊆ f.idealRange :=
  LieSubmodule.subset_lieSpan


theorem map_le_idealRange : I.map f ≤ f.idealRange := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    ⊢ LE.le (LieIdeal.map f I) f.idealRange
  -/
  rw [f.idealRange_eq_map]
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    ⊢ LE.le (LieIdeal.map f I) (LieIdeal.map f Top.top)
  -/
  exact LieIdeal.map_mono le_top
  /-
    🎉 no goals
  -/


theorem ker_le_comap : f.ker ≤ J.comap f :=
  LieIdeal.comap_mono bot_le


@[simp]
theorem ker_toSubmodule : LieSubmodule.toSubmodule (ker f) = LinearMap.ker (f : L →ₗ[R] L') :=
  rfl


@[deprecated (since := "2024-12-30")] alias ker_coeSubmodule := ker_toSubmodule


variable {f} in
@[simp]
theorem mem_ker {x : L} : x ∈ ker f ↔ f x = 0 :=
  show x ∈ LieSubmodule.toSubmodule (f.ker) ↔ _ by
    /-
      R : Type u
      L : Type v
      L' : Type w₂
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieRing L'
      inst✝¹ : LieAlgebra R L'
      inst✝ : LieAlgebra R L
      f : LieHom R L L'
      x : L
      ⊢ Iff (Membership.mem (↑f.ker) x) (Eq (f x) 0)
    -/
    simp only [ker_toSubmodule, LinearMap.mem_ker, coe_toLinearMap]
    /-
      🎉 no goals
    -/


theorem mem_idealRange (x : L) : f x ∈ idealRange f := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    x : L
    ⊢ Membership.mem f.idealRange (f x)
  -/
  rw [idealRange_eq_map]
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    x : L
    ⊢ Membership.mem (LieIdeal.map f Top.top) (f x)
  -/
  exact LieIdeal.mem_map (LieSubmodule.mem_top x)
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_idealRange_iff (h : IsIdealMorphism f) {y : L'} :
    y ∈ idealRange f ↔ ∃ x : L, f x = y := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    h : f.IsIdealMorphism
    y : L'
    ⊢ Iff (Membership.mem f.idealRange y) (Exists fun x => Eq (f x) y)
  -/
  rw [f.isIdealMorphism_def] at h
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    h : Eq (LieIdeal.toLieSubalgebra R L' f.idealRange) f.range
    y : L'
    ⊢ Iff (Membership.mem f.idealRange y) (Exists fun x => Eq (f x) y)
  -/
  rw [← LieSubmodule.mem_coe, ← LieIdeal.coe_toLieSubalgebra, h, f.range_coe, Set.mem_range]
  /-
    🎉 no goals
  -/


theorem le_ker_iff : I ≤ f.ker ↔ ∀ x, x ∈ I → f x = 0 := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    ⊢ Iff (LE.le I f.ker) (∀ (x : L), Membership.mem I x → Eq (f x) 0)
  -/
  constructor <;> intro h x hx
    /-
      case mp
      R : Type u
      L : Type v
      L' : Type w₂
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieRing L'
      inst✝¹ : LieAlgebra R L'
      inst✝ : LieAlgebra R L
      f : LieHom R L L'
      I : LieIdeal R L
      h : LE.le I f.ker
      x : L
      hx : Membership.mem I x
      ⊢ Eq (f x) 0
    -/
  · specialize h hx; rw [mem_ker] at h; exact h
                                        /-
                                          🎉 no goals
                                        -/
    /-
      case mpr
      R : Type u
      L : Type v
      L' : Type w₂
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieRing L'
      inst✝¹ : LieAlgebra R L'
      inst✝ : LieAlgebra R L
      f : LieHom R L L'
      I : LieIdeal R L
      h : ∀ (x : L), Membership.mem I x → Eq (f x) 0
      x : L
      hx : Membership.mem I x
      ⊢ Membership.mem f.ker x
    -/
  · rw [mem_ker]; apply h x hx
                  /-
                    🎉 no goals
                  -/


theorem ker_eq_bot : f.ker = ⊥ ↔ Function.Injective f := by
  rw [← LieSubmodule.toSubmodule_inj, ker_toSubmodule, LieSubmodule.bot_toSubmodule,
    LinearMap.ker_eq_bot, coe_toLinearMap]


@[simp]
theorem range_toSubmodule : (f.range : Submodule R L') = LinearMap.range (f : L →ₗ[R] L') :=
  rfl


@[deprecated (since := "2024-12-30")] alias range_coeSubmodule := range_toSubmodule


theorem range_eq_top : f.range = ⊤ ↔ Function.Surjective f := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    ⊢ Iff (Eq f.range Top.top) (Function.Surjective ⇑f)
  -/
  rw [← LieSubalgebra.toSubmodule_inj, range_toSubmodule, LieSubalgebra.top_toSubmodule]
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    ⊢ Iff (Eq (LinearMap.range ↑f) Top.top) (Function.Surjective ⇑f)
  -/
  exact LinearMap.range_eq_top
  /-
    🎉 no goals
  -/


@[simp]
theorem idealRange_eq_top_of_surjective (h : Function.Surjective f) : f.idealRange = ⊤ := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    h : Function.Surjective ⇑f
    ⊢ Eq f.idealRange Top.top
  -/
  rw [← f.range_eq_top] at h
  rw [idealRange_eq_lieSpan_range, h, ← LieSubalgebra.coe_toSubmodule, ←
    LieSubmodule.toSubmodule_inj, LieSubmodule.top_toSubmodule,
    LieSubalgebra.top_toSubmodule, LieSubmodule.coe_lieSpan_submodule_eq_iff]
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    h : Eq f.range Top.top
    ⊢ Exists fun N => Eq (↑N) Top.top
  -/
  use ⊤
  /-
    case h
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    h : Eq f.range Top.top
    ⊢ Eq (↑Top.top) Top.top
  -/
  exact LieSubmodule.top_toSubmodule
  /-
    🎉 no goals
  -/


theorem isIdealMorphism_of_surjective (h : Function.Surjective f) : f.IsIdealMorphism := by
  rw [isIdealMorphism_def, f.idealRange_eq_top_of_surjective h, f.range_eq_top.mpr h,
    LieIdeal.top_toLieSubalgebra]


@[simp]
theorem map_eq_bot_iff : I.map f = ⊥ ↔ I ≤ f.ker := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    ⊢ Iff (Eq (LieIdeal.map f I) Bot.bot) (LE.le I f.ker)
  -/
  rw [← le_bot_iff]
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    ⊢ Iff (LE.le (LieIdeal.map f I) Bot.bot) (LE.le I f.ker)
  -/
  exact LieIdeal.map_le_iff_le_comap
  /-
    🎉 no goals
  -/


theorem coe_map_of_surjective (h : Function.Surjective f) :
    LieSubmodule.toSubmodule (I.map f) = (LieSubmodule.toSubmodule I).map (f : L →ₗ[R] L') := by
  let J : LieIdeal R L' :=
    { (I : Submodule R L).map (f : L →ₗ[R] L') with
      lie_mem := fun {x y} hy ↦ by
        have hy' : ∃ x : L, x ∈ I ∧ f x = y := by simpa [hy]
        obtain ⟨z₂, hz₂, rfl⟩ := hy'
        obtain ⟨z₁, rfl⟩ := h x
        simp only [LieHom.coe_toLinearMap, SetLike.mem_coe, Set.mem_image,
          LieSubmodule.mem_toSubmodule, Submodule.mem_carrier, Submodule.map_coe]
        use ⁅z₁, z₂⁆
        exact ⟨I.lie_mem hz₂, f.map_lie z₁ z₂⟩ }
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    h : Function.Surjective ⇑f
    J : LieIdeal R L' :=
      let __src := Submodule.map (↑f) (LieIdeal.toLieSubalgebra R L I).toSubmodule;
      { toSubmodule := __src, lie_mem := ⋯ }
    ⊢ Eq (↑(LieIdeal.map f I)) (Submodule.map ↑f ↑I)
  -/
  erw [LieSubmodule.coe_lieSpan_submodule_eq_iff]
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    h : Function.Surjective ⇑f
    J : LieIdeal R L' :=
      let __src := Submodule.map (↑f) (LieIdeal.toLieSubalgebra R L I).toSubmodule;
      { toSubmodule := __src, lie_mem := ⋯ }
    ⊢ Exists fun N => Eq (↑N) (Submodule.map (↑f) (LieIdeal.toLieSubalgebra R L I) …
  -/
  use J
  /-
    🎉 no goals
  -/


theorem mem_map_of_surjective {y : L'} (h₁ : Function.Surjective f) (h₂ : y ∈ I.map f) :
    ∃ x : I, f x = y := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    y : L'
    h₁ : Function.Surjective ⇑f
    h₂ : Membership.mem (LieIdeal.map f I) y
    ⊢ Exists fun x => Eq (f ↑x) y
  -/
  rw [← LieSubmodule.mem_toSubmodule, coe_map_of_surjective h₁, Submodule.mem_map] at h₂
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    y : L'
    h₁ : Function.Surjective ⇑f
    h₂ : Exists fun y_1 => And (Membership.mem (↑I) y_1) (Eq (↑f y_1) y)
    ⊢ Exists fun x => Eq (f ↑x) y
  -/
  obtain ⟨x, hx, rfl⟩ := h₂
  /-
    case intro.intro
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    h₁ : Function.Surjective ⇑f
    x : L
    hx : Membership.mem (↑I) x
    ⊢ Exists fun x_1 => Eq (f ↑x_1) (↑f x)
  -/
  use ⟨x, hx⟩
  /-
    case h
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    h₁ : Function.Surjective ⇑f
    x : L
    hx : Membership.mem (↑I) x
    ⊢ Eq (f ↑⟨x, hx⟩) (↑f x)
  -/
  rw [LieHom.coe_toLinearMap]
  /-
    🎉 no goals
  -/


theorem bot_of_map_eq_bot {I : LieIdeal R L} (h₁ : Function.Injective f) (h₂ : I.map f = ⊥) :
    I = ⊥ := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    h₁ : Function.Injective ⇑f
    h₂ : Eq (LieIdeal.map f I) Bot.bot
    ⊢ Eq I Bot.bot
  -/
  rw [← f.ker_eq_bot, LieHom.ker] at h₁
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    h₁ : Eq (LieIdeal.comap f Bot.bot) Bot.bot
    h₂ : Eq (LieIdeal.map f I) Bot.bot
    ⊢ Eq I Bot.bot
  -/
  rw [eq_bot_iff, map_le_iff_le_comap, h₁] at h₂
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    h₁ : Eq (LieIdeal.comap f Bot.bot) Bot.bot
    h₂ : LE.le I Bot.bot
    ⊢ Eq I Bot.bot
  -/
  rw [eq_bot_iff]; exact h₂
                   /-
                     🎉 no goals
                   -/


/-- Given two nested Lie ideals `I₁ ⊆ I₂`, the inclusion `I₁ ↪ I₂` is a morphism of Lie algebras. -/
def inclusion {I₁ I₂ : LieIdeal R L} (h : I₁ ≤ I₂) : I₁ →ₗ⁅R⁆ I₂ where
  __ := Submodule.inclusion (show I₁.toSubmodule ≤ I₂.toSubmodule from h)
  map_lie' := rfl


@[simp]
theorem coe_inclusion {I₁ I₂ : LieIdeal R L} (h : I₁ ≤ I₂) (x : I₁) : (inclusion h x : L) = x :=
  rfl


theorem inclusion_apply {I₁ I₂ : LieIdeal R L} (h : I₁ ≤ I₂) (x : I₁) :
    inclusion h x = ⟨x.1, h x.2⟩ :=
  rfl


theorem inclusion_injective {I₁ I₂ : LieIdeal R L} (h : I₁ ≤ I₂) :
    Function.Injective (inclusion h) :=
  fun x y ↦ by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I₁ I₂ : LieIdeal R L
    h : LE.le I₁ I₂
    x y : Subtype fun x => Membership.mem I₁ x
    ⊢ Eq ((LieIdeal.inclusion h) x) ((LieIdeal.inclusion h) y) → Eq x y
  -/
  simp only [inclusion_apply, imp_self, Subtype.mk_eq_mk, SetLike.coe_eq_coe]
  /-
    🎉 no goals
  -/

-- Porting note: LHS simplifies, so moved @[simp] to new theorem `map_sup_ker_eq_map'`

theorem map_sup_ker_eq_map : LieIdeal.map f (I ⊔ f.ker) = LieIdeal.map f I := by
  suffices LieIdeal.map f (I ⊔ f.ker) ≤ LieIdeal.map f I by
    exact le_antisymm this (LieIdeal.map_mono le_sup_left)
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    ⊢ LE.le (LieIdeal.map f (Max.max I f.ker)) (LieIdeal.map f I)
  -/
  apply LieSubmodule.lieSpan_mono
  /-
    case h
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    ⊢ HasSubset.Subset ↑(Submodule.map (↑f) (LieIdeal.toLieSubalgebra R L (Max.max …
  -/
  rintro x ⟨y, hy₁, hy₂⟩
  /-
    case h.intro.intro
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    x : L'
    y : L
    hy₁ : Membership.mem (↑(LieIdeal.toLieSubalgebra R L (Max.max I f.ker)).toSubm …
    hy₂ : Eq (↑f y) x
    ⊢ Membership.mem (↑(Submodule.map (↑f) (LieIdeal.toLieSubalgebra R L I).toSubm …
  -/
  rw [← hy₂]
  /-
    case h.intro.intro
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    x : L'
    y : L
    hy₁ : Membership.mem (↑(LieIdeal.toLieSubalgebra R L (Max.max I f.ker)).toSubm …
    hy₂ : Eq (↑f y) x
    ⊢ Membership.mem (↑(Submodule.map (↑f) (LieIdeal.toLieSubalgebra R L I).toSubm …
  -/
  erw [LieSubmodule.mem_sup] at hy₁
  /-
    case h.intro.intro
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    x : L'
    y : L
    hy₁ : Exists fun y_1 => And (Membership.mem I y_1) (Exists fun z => And (Membe …
    hy₂ : Eq (↑f y) x
    ⊢ Membership.mem (↑(Submodule.map (↑f) (LieIdeal.toLieSubalgebra R L I).toSubm …
  -/
  obtain ⟨z₁, hz₁, z₂, hz₂, hy⟩ := hy₁
  /-
    case h.intro.intro.intro.intro.intro.intro
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    x : L'
    y : L
    hy₂ : Eq (↑f y) x
    z₁ : L
    hz₁ : Membership.mem I z₁
    z₂ : L
    hz₂ : Membership.mem f.ker z₂
    hy : Eq (HAdd.hAdd z₁ z₂) y
    ⊢ Membership.mem (↑(Submodule.map (↑f) (LieIdeal.toLieSubalgebra R L I).toSubm …
  -/
  rw [← hy]
  /-
    case h.intro.intro.intro.intro.intro.intro
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    x : L'
    y : L
    hy₂ : Eq (↑f y) x
    z₁ : L
    hz₁ : Membership.mem I z₁
    z₂ : L
    hz₂ : Membership.mem f.ker z₂
    hy : Eq (HAdd.hAdd z₁ z₂) y
    ⊢ Membership.mem (↑(Submodule.map (↑f) (LieIdeal.toLieSubalgebra R L I).toSubm …
  -/
  rw [f.coe_toLinearMap, f.map_add, LieHom.mem_ker.mp hz₂, add_zero]; exact ⟨z₁, hz₁, rfl⟩
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
theorem map_sup_ker_eq_map' :
    LieIdeal.map f I ⊔ LieIdeal.map f (LieHom.ker f) = LieIdeal.map f I := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    I : LieIdeal R L
    ⊢ Eq (Max.max (LieIdeal.map f I) (LieIdeal.map f f.ker)) (LieIdeal.map f I)
  -/
  simpa using map_sup_ker_eq_map (f := f)
  /-
    🎉 no goals
  -/


@[simp]
theorem map_comap_eq (h : f.IsIdealMorphism) : map f (comap f J) = f.idealRange ⊓ J := by
  /-
    R : Type u
    L : Type v
    L' : Type w₂
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : LieRing L'
    inst✝¹ : LieAlgebra R L'
    inst✝ : LieAlgebra R L
    f : LieHom R L L'
    J : LieIdeal R L'
    h : f.IsIdealMorphism
    ⊢ Eq (LieIdeal.map f (LieIdeal.comap f J)) (Min.min f.idealRange J)
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      L : Type v
      L' : Type w₂
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieRing L'
      inst✝¹ : LieAlgebra R L'
      inst✝ : LieAlgebra R L
      f : LieHom R L L'
      J : LieIdeal R L'
      h : f.IsIdealMorphism
      ⊢ LE.le (LieIdeal.map f (LieIdeal.comap f J)) (Min.min f.idealRange J)
    -/
  · rw [le_inf_iff]; exact ⟨f.map_le_idealRange _, map_comap_le⟩
                     /-
                       🎉 no goals
                     -/
    /-
      case a
      R : Type u
      L : Type v
      L' : Type w₂
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieRing L'
      inst✝¹ : LieAlgebra R L'
      inst✝ : LieAlgebra R L
      f : LieHom R L L'
      J : LieIdeal R L'
      h : f.IsIdealMorphism
      ⊢ LE.le (Min.min f.idealRange J) (LieIdeal.map f (LieIdeal.comap f J))
    -/
  · rw [f.isIdealMorphism_def] at h
    /-
      case a
      R : Type u
      L : Type v
      L' : Type w₂
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieRing L'
      inst✝¹ : LieAlgebra R L'
      inst✝ : LieAlgebra R L
      f : LieHom R L L'
      J : LieIdeal R L'
      h : Eq (LieIdeal.toLieSubalgebra R L' f.idealRange) f.range
      ⊢ LE.le (Min.min f.idealRange J) (LieIdeal.map f (LieIdeal.comap f J))
    -/
    rw [← SetLike.coe_subset_coe, LieSubmodule.inf_coe, ← coe_toLieSubalgebra, h]
    /-
      case a
      R : Type u
      L : Type v
      L' : Type w₂
      inst✝⁴ : CommRing R
      inst✝³ : LieRing L
      inst✝² : LieRing L'
      inst✝¹ : LieAlgebra R L'
      inst✝ : LieAlgebra R L
      f : LieHom R L L'
      J : LieIdeal R L'
      h : Eq (LieIdeal.toLieSubalgebra R L' f.idealRange) f.range
      ⊢ HasSubset.Subset (Inter.inter ↑f.range ↑J) ↑(LieIdeal.map f (LieIdeal.comap  …
    -/
    rintro y ⟨⟨x, h₁⟩, h₂⟩; rw [← h₁] at h₂ ⊢; exact mem_map h₂
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem comap_map_eq (h : ↑(map f I) = f '' I) : comap f (map f I) = I ⊔ f.ker := by
  rw [← LieSubmodule.toSubmodule_inj, comap_toSubmodule, I.map_toSubmodule f h,
    LieSubmodule.sup_toSubmodule, f.ker_toSubmodule, Submodule.comap_map_eq]


/-- Regarding an ideal `I` as a subalgebra, the inclusion map into its ambient space is a morphism
of Lie algebras. -/
def incl : I →ₗ⁅R⁆ L :=
  (I : LieSubalgebra R L).incl


@[simp]
theorem incl_range : I.incl.range = I :=
  (I : LieSubalgebra R L).incl_range


@[simp]
theorem incl_apply (x : I) : I.incl x = x :=
  rfl


@[simp]
theorem incl_coe : (I.incl.toLinearMap : I →ₗ[R] L) = (I : Submodule R L).subtype :=
  rfl


lemma incl_injective (I : LieIdeal R L) : Function.Injective I.incl :=
  Subtype.val_injective


@[simp]
                                                   /-
                                                     R : Type u
                                                     L : Type v
                                                     inst✝² : CommRing R
                                                     inst✝¹ : LieRing L
                                                     inst✝ : LieAlgebra R L
                                                     I : LieIdeal R L
                                                     ⊢ Eq (LieIdeal.comap I.incl I) Top.top
                                                   -/
theorem comap_incl_self : comap I.incl I = ⊤ := by ext; simp
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
                                        /-
                                          R : Type u
                                          L : Type v
                                          inst✝² : CommRing R
                                          inst✝¹ : LieRing L
                                          inst✝ : LieAlgebra R L
                                          I : LieIdeal R L
                                          ⊢ Eq I.incl.ker Bot.bot
                                        -/
theorem ker_incl : I.incl.ker = ⊥ := by ext; simp
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem incl_idealRange : I.incl.idealRange = I := by
  rw [LieHom.idealRange_eq_lieSpan_range, ← LieSubalgebra.coe_toSubmodule, ←
    LieSubmodule.toSubmodule_inj, incl_range, toLieSubalgebra_toSubmodule,
    LieSubmodule.coe_lieSpan_submodule_eq_iff]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    ⊢ Exists fun N => Eq ↑N ↑I
  -/
  use I
  /-
    🎉 no goals
  -/


theorem incl_isIdealMorphism : I.incl.IsIdealMorphism := by
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    ⊢ I.incl.IsIdealMorphism
  -/
  rw [I.incl.isIdealMorphism_def, incl_idealRange]
  /-
    R : Type u
    L : Type v
    inst✝² : CommRing R
    inst✝¹ : LieRing L
    inst✝ : LieAlgebra R L
    I : LieIdeal R L
    ⊢ Eq (LieIdeal.toLieSubalgebra R L I) I.incl.range
  -/
  exact (I : LieSubalgebra R L).incl_range.symm
  /-
    🎉 no goals
  -/


/-- The kernel of a morphism of Lie algebras, as an ideal in the domain. -/
def ker : LieSubmodule R L M :=
  LieSubmodule.comap f ⊥


@[simp]
theorem ker_toSubmodule : (f.ker : Submodule R M) = LinearMap.ker (f : M →ₗ[R] N) :=
  rfl


@[simp]
theorem mem_ker {m : M} : m ∈ f.ker ↔ f m = 0 :=
  Iff.rfl


@[simp]
theorem ker_id : (LieModuleHom.id : M →ₗ⁅R,L⁆ M).ker = ⊥ :=
  rfl


@[simp]
                                                    /-
                                                      R : Type u
                                                      L : Type v
                                                      M : Type w
                                                      N : Type w₁
                                                      inst✝⁷ : CommRing R
                                                      inst✝⁶ : LieRing L
                                                      inst✝⁵ : AddCommGroup M
                                                      inst✝⁴ : Module R M
                                                      inst✝³ : LieRingModule L M
                                                      inst✝² : AddCommGroup N
                                                      inst✝¹ : Module R N
                                                      inst✝ : LieRingModule L N
                                                      f : LieModuleHom R L M N
                                                      ⊢ Eq (f.comp f.ker.incl) 0
                                                    -/
theorem comp_ker_incl : f.comp f.ker.incl = 0 := by ext ⟨m, hm⟩; exact mem_ker.mp hm
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem le_ker_iff_map (M' : LieSubmodule R L M) : M' ≤ f.ker ↔ LieSubmodule.map f M' = ⊥ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    N : Type w₁
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : LieRingModule L N
    f : LieModuleHom R L M N
    M' : LieSubmodule R L M
    ⊢ Iff (LE.le M' f.ker) (Eq (LieSubmodule.map f M') Bot.bot)
  -/
  rw [ker, eq_bot_iff, LieSubmodule.map_le_iff_le_comap]
  /-
    🎉 no goals
  -/


/-- The range of a morphism of Lie modules `f : M → N` is a Lie submodule of `N`.
See Note [range copy pattern]. -/
def range : LieSubmodule R L N :=
  (LieSubmodule.map f ⊤).copy (Set.range f) Set.image_univ.symm


@[simp]
theorem coe_range : f.range = Set.range f :=
  rfl


@[simp]
theorem toSubmodule_range : f.range = LinearMap.range (f : M →ₗ[R] N) :=
  rfl


@[deprecated (since := "2024-12-30")] alias coeSubmodule_range := toSubmodule_range


@[simp]
theorem mem_range (n : N) : n ∈ f.range ↔ ∃ m, f m = n :=
  Iff.rfl


@[simp]
                                                       /-
                                                         R : Type u
                                                         L : Type v
                                                         M : Type w
                                                         N : Type w₁
                                                         inst✝⁷ : CommRing R
                                                         inst✝⁶ : LieRing L
                                                         inst✝⁵ : AddCommGroup M
                                                         inst✝⁴ : Module R M
                                                         inst✝³ : LieRingModule L M
                                                         inst✝² : AddCommGroup N
                                                         inst✝¹ : Module R N
                                                         inst✝ : LieRingModule L N
                                                         f : LieModuleHom R L M N
                                                         ⊢ Eq (LieSubmodule.map f Top.top) f.range
                                                       -/
theorem map_top : LieSubmodule.map f ⊤ = f.range := by ext; simp [LieSubmodule.mem_map]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem range_eq_top : f.range = ⊤ ↔ Function.Surjective f := by
  /-
    R : Type u
    L : Type v
    M : Type w
    N : Type w₁
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    inst✝ : LieRingModule L N
    f : LieModuleHom R L M N
    ⊢ Iff (Eq f.range Top.top) (Function.Surjective ⇑f)
  -/
  rw [SetLike.ext'_iff, coe_range, LieSubmodule.top_coe, Set.range_eq_univ]
  /-
    🎉 no goals
  -/


/-- A morphism of Lie modules `f : M → N` whose values lie in a Lie submodule `P ⊆ N` can be
restricted to a morphism of Lie modules `M → P`. -/
def codRestrict (P : LieSubmodule R L N) (f : M →ₗ⁅R,L⁆ N) (h : ∀ m, f m ∈ P) :
    M →ₗ⁅R,L⁆ P where
  toFun := f.toLinearMap.codRestrict P h
  __ := f.toLinearMap.codRestrict P h
                       /-
                         R : Type u
                         L : Type v
                         M : Type w
                         N : Type w₁
                         inst✝⁷ : CommRing R
                         inst✝⁶ : LieRing L
                         inst✝⁵ : AddCommGroup M
                         inst✝⁴ : Module R M
                         inst✝³ : LieRingModule L M
                         inst✝² : AddCommGroup N
                         inst✝¹ : Module R N
                         inst✝ : LieRingModule L N
                         f✝ : LieModuleHom R L M N
                         P : LieSubmodule R L N
                         f : LieModuleHom R L M N
                         h : ∀ (m : M), Membership.mem P (f m)
                         x : L
                         m : M
                         ⊢ Eq ({ toFun := ⇑(LinearMap.codRestrict (↑P) (↑f) h), map_add' := ⋯, map_smul …
                       -/
  map_lie' {x m} := by ext; simp
                            /-
                              🎉 no goals
                            -/


@[simp]
lemma codRestrict_apply (P : LieSubmodule R L N) (f : M →ₗ⁅R,L⁆ N) (h : ∀ m, f m ∈ P) (m : M) :
    (f.codRestrict P h m : N) = f m :=
  rfl


@[simp]
theorem ker_incl : N.incl.ker = ⊥ := (LieModuleHom.ker_eq_bot N.incl).mpr <| injective_incl N


@[simp]
theorem range_incl : N.incl.range = N := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N : LieSubmodule R L M
    ⊢ Eq N.incl.range N
  -/
  simp only [← toSubmodule_inj, LieModuleHom.toSubmodule_range, incl_coe]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N : LieSubmodule R L M
    ⊢ Eq (LinearMap.range (↑N).subtype) ↑N
  -/
  rw [Submodule.range_subtype]
  /-
    🎉 no goals
  -/


@[simp]
theorem comap_incl_self : comap N.incl N = ⊤ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N : LieSubmodule R L M
    ⊢ Eq (LieSubmodule.comap N.incl N) Top.top
  -/
  simp only [← toSubmodule_inj, toSubmodule_comap, incl_coe, top_toSubmodule]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N : LieSubmodule R L M
    ⊢ Eq (Submodule.comap (↑N).subtype ↑N) Top.top
  -/
  rw [Submodule.comap_subtype_self]
  /-
    🎉 no goals
  -/


                                                                     /-
                                                                       R : Type u
                                                                       L : Type v
                                                                       M : Type w
                                                                       inst✝⁴ : CommRing R
                                                                       inst✝³ : LieRing L
                                                                       inst✝² : AddCommGroup M
                                                                       inst✝¹ : Module R M
                                                                       inst✝ : LieRingModule L M
                                                                       N : LieSubmodule R L M
                                                                       ⊢ Eq (LieSubmodule.map N.incl Top.top) N
                                                                     -/
theorem map_incl_top : (⊤ : LieSubmodule R L N).map N.incl = N := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simp]
lemma map_le_range {M' : Type*}
    [AddCommGroup M'] [Module R M'] [LieRingModule L M'] (f : M →ₗ⁅R,L⁆ M') :
    N.map f ≤ f.range := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    N : LieSubmodule R L M
    M' : Type u_1
    inst✝² : AddCommGroup M'
    inst✝¹ : Module R M'
    inst✝ : LieRingModule L M'
    f : LieModuleHom R L M M'
    ⊢ LE.le (LieSubmodule.map f N) f.range
  -/
  rw [← LieModuleHom.map_top]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    N : LieSubmodule R L M
    M' : Type u_1
    inst✝² : AddCommGroup M'
    inst✝¹ : Module R M'
    inst✝ : LieRingModule L M'
    f : LieModuleHom R L M M'
    ⊢ LE.le (LieSubmodule.map f N) (LieSubmodule.map f Top.top)
  -/
  exact LieSubmodule.map_mono le_top
  /-
    🎉 no goals
  -/


@[simp]
lemma map_incl_lt_iff_lt_top {N' : LieSubmodule R L N} :
    N'.map (LieSubmodule.incl N) < N ↔ N' < ⊤ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N : LieSubmodule R L M
    N' : LieSubmodule R L (Subtype fun x => Membership.mem N x)
    ⊢ Iff (LT.lt (LieSubmodule.map N.incl N') N) (LT.lt N' Top.top)
  -/
  convert (LieSubmodule.mapOrderEmbedding (f := N.incl) Subtype.coe_injective).lt_iff_lt
  /-
    case h.e'_1.h.e'_4
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N : LieSubmodule R L M
    N' : LieSubmodule R L (Subtype fun x => Membership.mem N x)
    ⊢ Eq N ((LieSubmodule.mapOrderEmbedding ⋯) Top.top)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma map_incl_le {N' : LieSubmodule R L N} :
    N'.map N.incl ≤ N := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N : LieSubmodule R L M
    N' : LieSubmodule R L (Subtype fun x => Membership.mem N x)
    ⊢ LE.le (LieSubmodule.map N.incl N') N
  -/
  conv_rhs => rw [← N.map_incl_top]
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁴ : CommRing R
    inst✝³ : LieRing L
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : LieRingModule L M
    N : LieSubmodule R L M
    N' : LieSubmodule R L (Subtype fun x => Membership.mem N x)
    ⊢ LE.le (LieSubmodule.map N.incl N') (LieSubmodule.map N.incl Top.top)
  -/
  exact LieSubmodule.map_mono le_top
  /-
    🎉 no goals
  -/


/-- The natural equivalence between the 'top' Lie submodule and the enclosing Lie module. -/
def LieModuleEquiv.ofTop : (⊤ : LieSubmodule R L M) ≃ₗ⁅R,L⁆ M :=
  { LinearEquiv.ofTop ⊤ rfl with
    map_lie' := rfl }


@[simp, nolint simpNF] lemma LieModuleEquiv.ofTop_apply (x : (⊤ : LieSubmodule R L M)) :
    LieModuleEquiv.ofTop R L M x = x :=
  rfl


@[simp] lemma LieModuleEquiv.range_coe {M' : Type*}
    [AddCommGroup M'] [Module R M'] [LieRingModule L M'] (e : M ≃ₗ⁅R,L⁆ M') :
    LieModuleHom.range (e : M →ₗ⁅R,L⁆ M') = ⊤ := by
  /-
    R : Type u
    L : Type v
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    M : Type u_1
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    M' : Type u_2
    inst✝² : AddCommGroup M'
    inst✝¹ : Module R M'
    inst✝ : LieRingModule L M'
    e : LieModuleEquiv R L M M'
    ⊢ Eq e.range Top.top
  -/
  rw [LieModuleHom.range_eq_top]
  /-
    R : Type u
    L : Type v
    inst✝⁷ : CommRing R
    inst✝⁶ : LieRing L
    M : Type u_1
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : LieRingModule L M
    M' : Type u_2
    inst✝² : AddCommGroup M'
    inst✝¹ : Module R M'
    inst✝ : LieRingModule L M'
    e : LieModuleEquiv R L M M'
    ⊢ Function.Surjective ⇑e.toLieModuleHom
  -/
  exact e.surjective
  /-
    🎉 no goals
  -/


/-- The natural equivalence between the 'top' Lie subalgebra and the enclosing Lie algebra.

This is the Lie subalgebra version of `Submodule.topEquiv`. -/
def LieSubalgebra.topEquiv : (⊤ : LieSubalgebra R L) ≃ₗ⁅R⁆ L :=
  { (⊤ : LieSubalgebra R L).incl with
    invFun := fun x ↦ ⟨x, Set.mem_univ x⟩
                           /-
                             R : Type u
                             L : Type v
                             inst✝⁶ : CommRing R
                             inst✝⁵ : LieRing L
                             M : Type u_1
                             inst✝⁴ : AddCommGroup M
                             inst✝³ : Module R M
                             inst✝² : LieRingModule L M
                             inst✝¹ : LieAlgebra R L
                             inst✝ : LieModule R L M
                             x : Subtype fun x => Membership.mem Top.top x
                             ⊢ Eq ((fun x => ⟨x, ⋯⟩) ((↑__src✝).toFun x)) x
                           -/
    left_inv := fun x ↦ by ext; rfl
                                /-
                                  🎉 no goals
                                -/
    right_inv := fun _ ↦ rfl }


@[simp]
theorem LieSubalgebra.topEquiv_apply (x : (⊤ : LieSubalgebra R L)) : LieSubalgebra.topEquiv x = x :=
  rfl


/-- The natural equivalence between the 'top' Lie ideal and the enclosing Lie algebra.

This is the Lie ideal version of `Submodule.topEquiv`. -/
def LieIdeal.topEquiv : (⊤ : LieIdeal R L) ≃ₗ⁅R⁆ L :=
  LieSubalgebra.topEquiv

-- This lemma has always been bad, but https://github.com/leanprover/lean4/pull/2644 made `simp` start noticing

@[simp, nolint simpNF]
theorem LieIdeal.topEquiv_apply (x : (⊤ : LieIdeal R L)) : LieIdeal.topEquiv x = x :=
  rfl



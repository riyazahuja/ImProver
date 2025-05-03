/-- The type of Dirichlet characters of level `n`. -/
abbrev DirichletCharacter (R : Type*) [CommMonoidWithZero R] (n : ℕ) := MulChar (ZMod n) R


                                                                                        /-
                                                                                          R : Type u_1
                                                                                          inst✝ : CommMonoidWithZero R
                                                                                          n : Nat
                                                                                          χ : DirichletCharacter R n
                                                                                          a : ZMod n
                                                                                          ha : IsUnit a
                                                                                          ⊢ Eq (χ a) ↑((MulChar.toUnitHom χ) ha.unit)
                                                                                        -/
lemma toUnitHom_eq_char' {a : ZMod n} (ha : IsUnit a) : χ a = χ.toUnitHom ha.unit := by simp
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


                                                                                           /-
                                                                                             R : Type u_1
                                                                                             inst✝ : CommMonoidWithZero R
                                                                                             n : Nat
                                                                                             χ ψ : DirichletCharacter R n
                                                                                             ⊢ Iff (Eq (MulChar.toUnitHom χ) (MulChar.toUnitHom ψ)) (Eq χ ψ)
                                                                                           -/
lemma toUnitHom_inj (ψ : DirichletCharacter R n) : toUnitHom χ = toUnitHom ψ ↔ χ = ψ := by simp
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


@[deprecated (since := "2024-12-29")] alias toUnitHom_eq_iff := toUnitHom_inj


                                                               /-
                                                                 R : Type u_1
                                                                 inst✝ : CommMonoidWithZero R
                                                                 n : Nat
                                                                 χ : DirichletCharacter R n
                                                                 x : ZMod n
                                                                 ⊢ Eq (χ (HSub.hSub (↑n) x)) (χ (Neg.neg x))
                                                               -/
lemma eval_modulus_sub (x : ZMod n) : χ (n - x) = χ (-x) := by simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- A function that modifies the level of a Dirichlet character to some multiple
  of its original level. -/
noncomputable def changeLevel {n m : ℕ} (hm : n ∣ m) :
    DirichletCharacter R n →* DirichletCharacter R m where
  toFun ψ := MulChar.ofUnitHom (ψ.toUnitHom.comp (ZMod.unitsMap hm))
                 /-
                   R : Type u_1
                   inst✝ : CommMonoidWithZero R
                   n✝ : Nat
                   χ : DirichletCharacter R n✝
                   n m : Nat
                   hm : Dvd.dvd n m
                   ⊢ Eq ((fun ψ => MulChar.ofUnitHom ((MulChar.toUnitHom ψ).comp (ZMod.unitsMap h …
                 -/
  map_one' := by ext; simp
                      /-
                        🎉 no goals
                      -/
                       /-
                         R : Type u_1
                         inst✝ : CommMonoidWithZero R
                         n✝ : Nat
                         χ : DirichletCharacter R n✝
                         n m : Nat
                         hm : Dvd.dvd n m
                         ψ₁ ψ₂ : DirichletCharacter R n
                         ⊢ Eq ({ toFun := fun ψ => MulChar.ofUnitHom ((MulChar.toUnitHom ψ).comp (ZMod. …
                       -/
  map_mul' ψ₁ ψ₂ := by ext; simp
                            /-
                              🎉 no goals
                            -/


lemma changeLevel_def {m : ℕ} (hm : n ∣ m) :
    changeLevel hm χ = MulChar.ofUnitHom (χ.toUnitHom.comp (ZMod.unitsMap hm)) := rfl


lemma changeLevel_toUnitHom {m : ℕ} (hm : n ∣ m) :
    (changeLevel hm χ).toUnitHom = χ.toUnitHom.comp (ZMod.unitsMap hm) := by
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    n : Nat
    χ : DirichletCharacter R n
    m : Nat
    hm : Dvd.dvd n m
    ⊢ Eq (MulChar.toUnitHom ((DirichletCharacter.changeLevel hm) χ)) ((MulChar.toU …
  -/
  simp [changeLevel]
  /-
    🎉 no goals
  -/


/-- The `changeLevel` map is injective (except in the degenerate case `m = 0`). -/
lemma changeLevel_injective {m : ℕ} [NeZero m] (hm : n ∣ m) :
    Function.Injective (changeLevel (R := R) hm) := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    n m : Nat
    inst✝ : NeZero m
    hm : Dvd.dvd n m
    ⊢ Function.Injective ⇑(DirichletCharacter.changeLevel hm)
  -/
  intro _ _ h
  /-
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    n m : Nat
    inst✝ : NeZero m
    hm : Dvd.dvd n m
    a₁✝ a₂✝ : DirichletCharacter R n
    h : Eq ((DirichletCharacter.changeLevel hm) a₁✝) ((DirichletCharacter.changeLe …
    ⊢ Eq a₁✝ a₂✝
  -/
  ext1 y
  /-
    case h
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    n m : Nat
    inst✝ : NeZero m
    hm : Dvd.dvd n m
    a₁✝ a₂✝ : DirichletCharacter R n
    h : Eq ((DirichletCharacter.changeLevel hm) a₁✝) ((DirichletCharacter.changeLe …
    y : Units (ZMod n)
    ⊢ Eq (a₁✝ ↑y) (a₂✝ ↑y)
  -/
  obtain ⟨z, rfl⟩ := ZMod.unitsMap_surjective hm y
  /-
    case h.intro
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    n m : Nat
    inst✝ : NeZero m
    hm : Dvd.dvd n m
    a₁✝ a₂✝ : DirichletCharacter R n
    h : Eq ((DirichletCharacter.changeLevel hm) a₁✝) ((DirichletCharacter.changeLe …
    z : Units (ZMod m)
    ⊢ Eq (a₁✝ ↑((ZMod.unitsMap hm) z)) (a₂✝ ↑((ZMod.unitsMap hm) z))
  -/
  rw [MulChar.ext_iff] at h
  /-
    case h.intro
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    n m : Nat
    inst✝ : NeZero m
    hm : Dvd.dvd n m
    a₁✝ a₂✝ : DirichletCharacter R n
    h : ∀ (a : Units (ZMod m)), Eq (((DirichletCharacter.changeLevel hm) a₁✝) ↑a)  …
    z : Units (ZMod m)
    ⊢ Eq (a₁✝ ↑((ZMod.unitsMap hm) z)) (a₂✝ ↑((ZMod.unitsMap hm) z))
  -/
  simpa [changeLevel_def] using h z
  /-
    🎉 no goals
  -/


@[simp]
lemma changeLevel_eq_one_iff {m : ℕ} {χ : DirichletCharacter R n} (hm : n ∣ m) [NeZero m] :
    changeLevel hm χ = 1 ↔ χ = 1 :=
  map_eq_one_iff _ (changeLevel_injective hm)


@[simp]
lemma changeLevel_self : changeLevel (dvd_refl n) χ = χ := by
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    n : Nat
    χ : DirichletCharacter R n
    ⊢ Eq ((DirichletCharacter.changeLevel ⋯) χ) χ
  -/
  simp [changeLevel, ZMod.unitsMap]
  /-
    🎉 no goals
  -/


lemma changeLevel_self_toUnitHom : (changeLevel (dvd_refl n) χ).toUnitHom = χ.toUnitHom := by
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    n : Nat
    χ : DirichletCharacter R n
    ⊢ Eq (MulChar.toUnitHom ((DirichletCharacter.changeLevel ⋯) χ)) (MulChar.toUni …
  -/
  rw [changeLevel_self]
  /-
    🎉 no goals
  -/


lemma changeLevel_trans {m d : ℕ} (hm : n ∣ m) (hd : m ∣ d) :
    changeLevel (dvd_trans hm hd) χ = changeLevel hd (changeLevel hm χ) := by
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    n : Nat
    χ : DirichletCharacter R n
    m d : Nat
    hm : Dvd.dvd n m
    hd : Dvd.dvd m d
    ⊢ Eq ((DirichletCharacter.changeLevel ⋯) χ) ((DirichletCharacter.changeLevel h …
  -/
  simp [changeLevel_def, MonoidHom.comp_assoc, ZMod.unitsMap_comp]
  /-
    🎉 no goals
  -/


lemma changeLevel_eq_cast_of_dvd {m : ℕ} (hm : n ∣ m) (a : Units (ZMod m)) :
    (changeLevel hm χ) a = χ (ZMod.cast (a : ZMod m)) := by
  set_option tactic.skipAssignedInstances false in
  simpa [changeLevel_def, Function.comp_apply, MonoidHom.coe_comp] using
      toUnitHom_eq_char' _ <| ZMod.isUnit_cast_of_dvd hm a


/-- `χ` of level `n` factors through a Dirichlet character `χ₀` of level `d` if `d ∣ n` and
`χ₀ = χ ∘ (ZMod n → ZMod d)`. -/
def FactorsThrough (d : ℕ) : Prop :=
  ∃ (h : d ∣ n) (χ₀ : DirichletCharacter R d), χ = changeLevel h χ₀


lemma changeLevel_factorsThrough {m : ℕ} (hm : n ∣ m) : FactorsThrough (changeLevel hm χ) n :=
  ⟨hm, χ, rfl⟩


/-- The fact that `d` divides `n` when `χ` factors through a Dirichlet character at level `d` -/
lemma dvd {d : ℕ} (h : FactorsThrough χ d) : d ∣ n := h.1


/-- The Dirichlet character at level `d` through which `χ` factors -/
noncomputable
def χ₀ {d : ℕ} (h : FactorsThrough χ d) : DirichletCharacter R d := Classical.choose h.2


/-- The fact that `χ` factors through `χ₀` of level `d` -/
lemma eq_changeLevel {d : ℕ} (h : FactorsThrough χ d) : χ = changeLevel h.dvd h.χ₀ :=
  Classical.choose_spec h.2


/-- The character of level `d` through which `χ` factors is uniquely determined. -/
lemma existsUnique {d : ℕ} [NeZero n] (h : FactorsThrough χ d) :
    ∃! χ' : DirichletCharacter R d, χ = changeLevel h.dvd χ' := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    n : Nat
    χ : DirichletCharacter R n
    d : Nat
    inst✝ : NeZero n
    h : χ.FactorsThrough d
    ⊢ ExistsUnique fun χ' => Eq χ ((DirichletCharacter.changeLevel ⋯) χ')
  -/
  rcases h with ⟨hd, χ₂, rfl⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    n d : Nat
    inst✝ : NeZero n
    hd : Dvd.dvd d n
    χ₂ : DirichletCharacter R d
    ⊢ ExistsUnique fun χ' => Eq ((DirichletCharacter.changeLevel hd) χ₂) ((Dirichl …
  -/
  exact ⟨χ₂, rfl, fun χ₃ hχ₃ ↦ (changeLevel_injective hd hχ₃).symm⟩
  /-
    🎉 no goals
  -/


variable (χ) in
lemma same_level : FactorsThrough χ n := ⟨dvd_refl n, χ, (changeLevel_self χ).symm⟩


variable {χ} in
/-- A Dirichlet character `χ` factors through `d | n` iff its associated unit-group hom is trivial
on the kernel of `ZMod.unitsMap`. -/
lemma factorsThrough_iff_ker_unitsMap {d : ℕ} [NeZero n] (hd : d ∣ n) :
    FactorsThrough χ d ↔ (ZMod.unitsMap hd).ker ≤ χ.toUnitHom.ker := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoidWithZero R
    n : Nat
    χ : DirichletCharacter R n
    d : Nat
    inst✝ : NeZero n
    hd : Dvd.dvd d n
    ⊢ Iff (χ.FactorsThrough d) (LE.le (ZMod.unitsMap hd).ker (MulChar.toUnitHom χ) …
  -/
  refine ⟨fun ⟨_, ⟨χ₀, hχ₀⟩⟩ x hx ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      R : Type u_1
      inst✝¹ : CommMonoidWithZero R
      n : Nat
      χ : DirichletCharacter R n
      d : Nat
      inst✝ : NeZero n
      hd : Dvd.dvd d n
      x✝ : χ.FactorsThrough d
      x : Units (ZMod n)
      hx : Membership.mem (ZMod.unitsMap hd).ker x
      w✝ : Dvd.dvd d n
      χ₀ : DirichletCharacter R d
      hχ₀ : Eq χ ((DirichletCharacter.changeLevel w✝) χ₀)
      ⊢ Membership.mem (MulChar.toUnitHom χ).ker x
    -/
  · rw [MonoidHom.mem_ker, hχ₀, changeLevel_toUnitHom, MonoidHom.comp_apply, hx, map_one]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝¹ : CommMonoidWithZero R
      n : Nat
      χ : DirichletCharacter R n
      d : Nat
      inst✝ : NeZero n
      hd : Dvd.dvd d n
      h : LE.le (ZMod.unitsMap hd).ker (MulChar.toUnitHom χ).ker
      ⊢ χ.FactorsThrough d
    -/
  · let E := MonoidHom.liftOfSurjective _ (ZMod.unitsMap_surjective hd) ⟨_, h⟩
    /-
      case refine_2
      R : Type u_1
      inst✝¹ : CommMonoidWithZero R
      n : Nat
      χ : DirichletCharacter R n
      d : Nat
      inst✝ : NeZero n
      hd : Dvd.dvd d n
      h : LE.le (ZMod.unitsMap hd).ker (MulChar.toUnitHom χ).ker
      E : MonoidHom (Units (ZMod d)) (Units R) := ((ZMod.unitsMap hd).liftOfSurjecti …
      ⊢ χ.FactorsThrough d
    -/
    have hE : E.comp (ZMod.unitsMap hd) = χ.toUnitHom := MonoidHom.liftOfRightInverse_comp ..
    /-
      case refine_2
      R : Type u_1
      inst✝¹ : CommMonoidWithZero R
      n : Nat
      χ : DirichletCharacter R n
      d : Nat
      inst✝ : NeZero n
      hd : Dvd.dvd d n
      h : LE.le (ZMod.unitsMap hd).ker (MulChar.toUnitHom χ).ker
      E : MonoidHom (Units (ZMod d)) (Units R) := ((ZMod.unitsMap hd).liftOfSurjecti …
      hE : Eq (E.comp (ZMod.unitsMap hd)) (MulChar.toUnitHom χ)
      ⊢ χ.FactorsThrough d
    -/
    refine ⟨hd, MulChar.ofUnitHom E, equivToUnitHom.injective (?_ : toUnitHom _ = toUnitHom _)⟩
    simp_rw [changeLevel_toUnitHom, toUnitHom_eq, ofUnitHom_eq, Equiv.apply_symm_apply, hE,
      toUnitHom_eq]


lemma level_one (χ : DirichletCharacter R 1) : χ = 1 := by
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    χ : DirichletCharacter R 1
    ⊢ Eq χ 1
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    χ : DirichletCharacter R 1
    a✝ : Units (ZMod 1)
    ⊢ Eq (χ ↑a✝) (1 ↑a✝)
  -/
  simp [units_eq_one]
  /-
    🎉 no goals
  -/


lemma level_one' (hn : n = 1) : χ = 1 := by
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    n : Nat
    χ : DirichletCharacter R n
    hn : Eq n 1
    ⊢ Eq χ 1
  -/
  subst hn
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    χ : DirichletCharacter R 1
    ⊢ Eq χ 1
  -/
  exact level_one _
  /-
    🎉 no goals
  -/


instance : Subsingleton (DirichletCharacter R 1) := by
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    n : Nat
    χ : DirichletCharacter R n
    ⊢ Subsingleton (DirichletCharacter R 1)
  -/
  refine subsingleton_iff.mpr (fun χ χ' ↦ ?_)
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    n : Nat
    χ✝ : DirichletCharacter R n
    χ χ' : DirichletCharacter R 1
    ⊢ Eq χ χ'
  -/
  simp [level_one]
  /-
    🎉 no goals
  -/


noncomputable instance : Unique (DirichletCharacter R 1) := Unique.mk' (DirichletCharacter R 1)


/-- A Dirichlet character of modulus `≠ 1` maps `0` to `0`. -/
lemma map_zero' (hn : n ≠ 1) : χ 0 = 0 :=
  have := ZMod.nontrivial_iff.mpr hn; χ.map_zero


lemma changeLevel_one {d : ℕ} (h : d ∣ n) :
    changeLevel h (1 : DirichletCharacter R d) = 1 := by
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    n d : Nat
    h : Dvd.dvd d n
    ⊢ Eq ((DirichletCharacter.changeLevel h) 1) 1
  -/
  simp
  /-
    🎉 no goals
  -/


lemma factorsThrough_one_iff : FactorsThrough χ 1 ↔ χ = 1 := by
  refine ⟨fun ⟨_, χ₀, hχ₀⟩ ↦ ?_,
          fun h ↦ ⟨one_dvd n, 1, by rw [h, changeLevel_one]⟩⟩
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    n : Nat
    χ : DirichletCharacter R n
    x✝ : χ.FactorsThrough 1
    w✝ : Dvd.dvd 1 n
    χ₀ : DirichletCharacter R 1
    hχ₀ : Eq χ ((DirichletCharacter.changeLevel w✝) χ₀)
    ⊢ Eq χ 1
  -/
  rwa [level_one χ₀, changeLevel_one] at hχ₀
  /-
    🎉 no goals
  -/


/-- The set of natural numbers `d` such that `χ` factors through a character of level `d`. -/
def conductorSet : Set ℕ := {d : ℕ | FactorsThrough χ d}


lemma mem_conductorSet_iff {x : ℕ} : x ∈ conductorSet χ ↔ FactorsThrough χ x := Iff.refl _


lemma level_mem_conductorSet : n ∈ conductorSet χ := FactorsThrough.same_level χ


lemma mem_conductorSet_dvd {x : ℕ} (hx : x ∈ conductorSet χ) : x ∣ n := hx.dvd


/-- The minimum natural number level `n` through which `χ` factors. -/
noncomputable def conductor : ℕ := sInf (conductorSet χ)


lemma conductor_mem_conductorSet : conductor χ ∈ conductorSet χ :=
  Nat.sInf_mem (Set.nonempty_of_mem (level_mem_conductorSet χ))


lemma conductor_dvd_level : conductor χ ∣ n := (conductor_mem_conductorSet χ).dvd


lemma factorsThrough_conductor : FactorsThrough χ (conductor χ) := conductor_mem_conductorSet χ


lemma conductor_ne_zero (hn : n ≠ 0) : conductor χ ≠ 0 :=
  fun h ↦ hn <| Nat.eq_zero_of_zero_dvd <| h ▸ conductor_dvd_level _


/-- The conductor of the trivial character is 1. -/
lemma conductor_one (hn : n ≠ 0) : conductor (1 : DirichletCharacter R n) = 1 := by
  suffices FactorsThrough (1 : DirichletCharacter R n) 1 by
    have h : conductor (1 : DirichletCharacter R n) ≤ 1 :=
      Nat.sInf_le <| (mem_conductorSet_iff _).mpr this
    exact Nat.le_antisymm h (Nat.pos_of_ne_zero <| conductor_ne_zero _ hn)
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    n : Nat
    hn : Ne n 0
    ⊢ DirichletCharacter.FactorsThrough 1 1
  -/
  exact (factorsThrough_one_iff _).mpr rfl
  /-
    🎉 no goals
  -/


lemma eq_one_iff_conductor_eq_one (hn : n ≠ 0) : χ = 1 ↔ conductor χ = 1 := by
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    n : Nat
    χ : DirichletCharacter R n
    hn : Ne n 0
    ⊢ Iff (Eq χ 1) (Eq χ.conductor 1)
  -/
  refine ⟨fun h ↦ h ▸ conductor_one hn, fun hχ ↦ ?_⟩
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    n : Nat
    χ : DirichletCharacter R n
    hn : Ne n 0
    hχ : Eq χ.conductor 1
    ⊢ Eq χ 1
  -/
  obtain ⟨h', χ₀, h⟩ := factorsThrough_conductor χ
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    n : Nat
    χ : DirichletCharacter R n
    hn : Ne n 0
    hχ : Eq χ.conductor 1
    h' : Dvd.dvd χ.conductor n
    χ₀ : DirichletCharacter R χ.conductor
    h : Eq χ ((DirichletCharacter.changeLevel h') χ₀)
    ⊢ Eq χ 1
  -/
  exact (level_one' χ₀ hχ ▸ h).trans <| changeLevel_one h'
  /-
    🎉 no goals
  -/


lemma conductor_eq_zero_iff_level_eq_zero : conductor χ = 0 ↔ n = 0 := by
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    n : Nat
    χ : DirichletCharacter R n
    ⊢ Iff (Eq χ.conductor 0) (Eq n 0)
  -/
  refine ⟨(conductor_ne_zero χ).mtr, ?_⟩
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    n : Nat
    χ : DirichletCharacter R n
    ⊢ Eq n 0 → Eq χ.conductor 0
  -/
  rintro rfl
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    χ : DirichletCharacter R 0
    ⊢ Eq χ.conductor 0
  -/
  exact Nat.sInf_eq_zero.mpr <| Or.inl <| level_mem_conductorSet χ
  /-
    🎉 no goals
  -/


lemma conductor_le_conductor_mem_conductorSet {d : ℕ} (hd : d ∈ conductorSet χ) :
    χ.conductor ≤ (Classical.choose hd.2).conductor := by
  refine Nat.sInf_le <| (mem_conductorSet_iff χ).mpr <|
    ⟨dvd_trans (conductor_dvd_level _) hd.1,
     (factorsThrough_conductor (Classical.choose hd.2)).2.choose, ?_⟩
  rw [changeLevel_trans _ (conductor_dvd_level _) hd.dvd,
      ← (factorsThrough_conductor (Classical.choose hd.2)).2.choose_spec]
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    n : Nat
    χ : DirichletCharacter R n
    d : Nat
    hd : Membership.mem χ.conductorSet d
    ⊢ Eq χ ((DirichletCharacter.changeLevel ⋯) (Classical.choose ⋯))
  -/
  exact hd.eq_changeLevel
  /-
    🎉 no goals
  -/


/-- A character is primitive if its level is equal to its conductor. -/
def IsPrimitive : Prop := conductor χ = n


@[deprecated (since := "2024-06-16")] alias isPrimitive := IsPrimitive


lemma isPrimitive_def : IsPrimitive χ ↔ conductor χ = n := Iff.rfl


lemma isPrimitive_one_level_one : IsPrimitive (1 : DirichletCharacter R 1) :=
  Nat.dvd_one.mp (conductor_dvd_level _)


lemma isPritive_one_level_zero : IsPrimitive (1 : DirichletCharacter R 0) :=
  conductor_eq_zero_iff_level_eq_zero.mpr rfl


lemma conductor_one_dvd (n : ℕ) : conductor (1 : DirichletCharacter R 1) ∣ n := by
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    n : Nat
    ⊢ Dvd.dvd (DirichletCharacter.conductor 1) n
  -/
  rw [(isPrimitive_def _).mp isPrimitive_one_level_one]
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    n : Nat
    ⊢ Dvd.dvd 1 n
  -/
  apply one_dvd _
  /-
    🎉 no goals
  -/


/-- The primitive character associated to a Dirichlet character. -/
noncomputable def primitiveCharacter : DirichletCharacter R χ.conductor :=
  Classical.choose (factorsThrough_conductor χ).choose_spec


lemma primitiveCharacter_isPrimitive : IsPrimitive (χ.primitiveCharacter) := by
  /-
    R : Type u_1
    inst✝ : CommMonoidWithZero R
    n : Nat
    χ : DirichletCharacter R n
    ⊢ χ.primitiveCharacter.IsPrimitive
  -/
  by_cases h : χ.conductor = 0
    /-
      case pos
      R : Type u_1
      inst✝ : CommMonoidWithZero R
      n : Nat
      χ : DirichletCharacter R n
      h : Eq χ.conductor 0
      ⊢ χ.primitiveCharacter.IsPrimitive
    -/
  · rw [isPrimitive_def]
    /-
      case pos
      R : Type u_1
      inst✝ : CommMonoidWithZero R
      n : Nat
      χ : DirichletCharacter R n
      h : Eq χ.conductor 0
      ⊢ Eq χ.primitiveCharacter.conductor χ.conductor
    -/
    convert conductor_eq_zero_iff_level_eq_zero.mpr h
    /-
      🎉 no goals
    -/
  · exact le_antisymm (Nat.le_of_dvd (Nat.pos_of_ne_zero h) (conductor_dvd_level _)) <|
      conductor_le_conductor_mem_conductorSet <| conductor_mem_conductorSet χ


lemma primitiveCharacter_one (hn : n ≠ 0) :
    (1 : DirichletCharacter R n).primitiveCharacter = 1 := by
  rw [eq_one_iff_conductor_eq_one <| (@conductor_one R _ _ hn) ▸ Nat.one_ne_zero,
      (isPrimitive_def _).1 (1 : DirichletCharacter R n).primitiveCharacter_isPrimitive,
      conductor_one hn]


/-- Dirichlet character associated to multiplication of Dirichlet characters,
after changing both levels to the same -/
noncomputable def mul {m : ℕ} (χ₁ : DirichletCharacter R n) (χ₂ : DirichletCharacter R m) :
    DirichletCharacter R (Nat.lcm n m) :=
  changeLevel (Nat.dvd_lcm_left n m) χ₁ * changeLevel (Nat.dvd_lcm_right n m) χ₂


/-- Primitive character associated to multiplication of Dirichlet characters,
after changing both levels to the same -/
noncomputable def primitive_mul {m : ℕ} (χ₁ : DirichletCharacter R n)
    (χ₂ : DirichletCharacter R m) : DirichletCharacter R (mul χ₁ χ₂).conductor :=
  primitiveCharacter (mul χ₁ χ₂)


lemma mul_def {n m : ℕ} {χ : DirichletCharacter R n} {ψ : DirichletCharacter R m} :
    χ.primitive_mul ψ = primitiveCharacter (mul χ ψ) :=
  rfl


lemma primitive_mul_isPrimitive {m : ℕ} (ψ : DirichletCharacter R m) :
    IsPrimitive (primitive_mul χ ψ) :=
  primitiveCharacter_isPrimitive _


@[deprecated (since := "2024-06-16")] alias isPrimitive.primitive_mul := primitive_mul_isPrimitive

/-
### Even and odd characters
-/


/-- A Dirichlet character is odd if its value at -1 is -1. -/
def Odd : Prop := ψ (-1) = -1


/-- A Dirichlet character is even if its value at -1 is 1. -/
def Even : Prop := ψ (-1) = 1


lemma even_or_odd [NoZeroDivisors S] : ψ.Even ∨ ψ.Odd := by
  /-
    S : Type u_2
    inst✝¹ : CommRing S
    m : Nat
    ψ : DirichletCharacter S m
    inst✝ : NoZeroDivisors S
    ⊢ Or ψ.Even ψ.Odd
  -/
  suffices ψ (-1) ^ 2 = 1 by convert sq_eq_one_iff.mp this
  /-
    S : Type u_2
    inst✝¹ : CommRing S
    m : Nat
    ψ : DirichletCharacter S m
    inst✝ : NoZeroDivisors S
    ⊢ Eq (HPow.hPow (ψ (-1)) 2) 1
  -/
  rw [← map_pow _, neg_one_sq, map_one]
  /-
    🎉 no goals
  -/


lemma not_even_and_odd [NeZero (2 : S)] : ¬(ψ.Even ∧ ψ.Odd) := by
  /-
    S : Type u_2
    inst✝¹ : CommRing S
    m : Nat
    ψ : DirichletCharacter S m
    inst✝ : NeZero 2
    ⊢ Not (And ψ.Even ψ.Odd)
  -/
  rintro ⟨(h : _ = 1), (h' : _ = -1)⟩
  /-
    case intro
    S : Type u_2
    inst✝¹ : CommRing S
    m : Nat
    ψ : DirichletCharacter S m
    inst✝ : NeZero 2
    h : Eq (ψ (-1)) 1
    h' : Eq (ψ (-1)) (-1)
    ⊢ False
  -/
  simp only [h', neg_eq_iff_add_eq_zero, one_add_one_eq_two, two_ne_zero] at h
  /-
    🎉 no goals
  -/


lemma Even.not_odd [NeZero (2 : S)] (hψ : Even ψ) : ¬Odd ψ :=
  not_and.mp ψ.not_even_and_odd hψ


lemma Odd.not_even [NeZero (2 : S)] (hψ : Odd ψ) : ¬Even ψ :=
  not_and'.mp ψ.not_even_and_odd hψ


lemma Odd.toUnitHom_eval_neg_one (hψ : ψ.Odd) : ψ.toUnitHom (-1) = -1 := by
  /-
    S : Type u_2
    inst✝ : CommRing S
    m : Nat
    ψ : DirichletCharacter S m
    hψ : ψ.Odd
    ⊢ Eq ((MulChar.toUnitHom ψ) (-1)) (-1)
  -/
  rw [← Units.eq_iff, MulChar.coe_toUnitHom]
  /-
    S : Type u_2
    inst✝ : CommRing S
    m : Nat
    ψ : DirichletCharacter S m
    hψ : ψ.Odd
    ⊢ Eq (ψ ↑(-1)) ↑(-1)
  -/
  exact hψ
  /-
    🎉 no goals
  -/


lemma Even.toUnitHom_eval_neg_one (hψ : ψ.Even) : ψ.toUnitHom (-1) = 1 := by
  /-
    S : Type u_2
    inst✝ : CommRing S
    m : Nat
    ψ : DirichletCharacter S m
    hψ : ψ.Even
    ⊢ Eq ((MulChar.toUnitHom ψ) (-1)) 1
  -/
  rw [← Units.eq_iff, MulChar.coe_toUnitHom]
  /-
    S : Type u_2
    inst✝ : CommRing S
    m : Nat
    ψ : DirichletCharacter S m
    hψ : ψ.Even
    ⊢ Eq (ψ ↑(-1)) ↑1
  -/
  exact hψ
  /-
    🎉 no goals
  -/


lemma Odd.eval_neg (x : ZMod m) (hψ : ψ.Odd) : ψ (- x) = - ψ x := by
  /-
    S : Type u_2
    inst✝ : CommRing S
    m : Nat
    ψ : DirichletCharacter S m
    x : ZMod m
    hψ : ψ.Odd
    ⊢ Eq (ψ (Neg.neg x)) (Neg.neg (ψ x))
  -/
  rw [Odd] at hψ
  /-
    S : Type u_2
    inst✝ : CommRing S
    m : Nat
    ψ : DirichletCharacter S m
    x : ZMod m
    hψ : Eq (ψ (-1)) (-1)
    ⊢ Eq (ψ (Neg.neg x)) (Neg.neg (ψ x))
  -/
  rw [← neg_one_mul, map_mul]
  /-
    S : Type u_2
    inst✝ : CommRing S
    m : Nat
    ψ : DirichletCharacter S m
    x : ZMod m
    hψ : Eq (ψ (-1)) (-1)
    ⊢ Eq (HMul.hMul (ψ (-1)) (ψ x)) (Neg.neg (ψ x))
  -/
  simp [hψ]
  /-
    🎉 no goals
  -/


lemma Even.eval_neg (x : ZMod m) (hψ : ψ.Even) : ψ (- x) = ψ x := by
  /-
    S : Type u_2
    inst✝ : CommRing S
    m : Nat
    ψ : DirichletCharacter S m
    x : ZMod m
    hψ : ψ.Even
    ⊢ Eq (ψ (Neg.neg x)) (ψ x)
  -/
  rw [Even] at hψ
  /-
    S : Type u_2
    inst✝ : CommRing S
    m : Nat
    ψ : DirichletCharacter S m
    x : ZMod m
    hψ : Eq (ψ (-1)) 1
    ⊢ Eq (ψ (Neg.neg x)) (ψ x)
  -/
  rw [← neg_one_mul, map_mul]
  /-
    S : Type u_2
    inst✝ : CommRing S
    m : Nat
    ψ : DirichletCharacter S m
    x : ZMod m
    hψ : Eq (ψ (-1)) 1
    ⊢ Eq (HMul.hMul (ψ (-1)) (ψ x)) (ψ x)
  -/
  simp [hψ]
  /-
    🎉 no goals
  -/


/-- An even Dirichlet character is an even function. -/
lemma Even.to_fun {χ : DirichletCharacter S m} (hχ : Even χ) : Function.Even χ :=
             /-
               S : Type u_2
               inst✝ : CommRing S
               m : Nat
               χ : DirichletCharacter S m
               hχ : χ.Even
               x✝ : ZMod m
               ⊢ Eq (χ (Neg.neg x✝)) (χ x✝)
             -/
  fun _ ↦ by rw [← neg_one_mul, map_mul, hχ, one_mul]
             /-
               🎉 no goals
             -/


/-- An odd Dirichlet character is an odd function. -/
lemma Odd.to_fun {χ : DirichletCharacter S m} (hχ : Odd χ) : Function.Odd χ :=
             /-
               S : Type u_2
               inst✝ : CommRing S
               m : Nat
               χ : DirichletCharacter S m
               hχ : χ.Odd
               x✝ : ZMod m
               ⊢ Eq (χ (Neg.neg x✝)) (Neg.neg (χ x✝))
             -/
  fun _ ↦ by rw [← neg_one_mul, map_mul, hχ, neg_one_mul]
             /-
               🎉 no goals
             -/



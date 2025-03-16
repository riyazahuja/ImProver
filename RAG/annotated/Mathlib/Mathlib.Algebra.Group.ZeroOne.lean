@[to_additive]
class One (α : Type u) where
  one : α


@[to_additive existing Zero.toOfNat0]
instance (priority := 300) One.toOfNat1 {α} [One α] : OfNat α (nat_lit 1) where
  ofNat := ‹One α›.1

@[to_additive existing Zero.ofOfNat0, to_additive_change_numeral 2]
instance (priority := 200) One.ofOfNat1 {α} [OfNat α (nat_lit 1)] : One α where
  one := 1


instance (priority := 20) Zero.instNonempty [Zero α] : Nonempty α :=
  ⟨0⟩


instance (priority := 20) One.instNonempty [One α] : Nonempty α :=
  ⟨1⟩


@[to_additive]
theorem Subsingleton.eq_one [One α] [Subsingleton α] (a : α) : a = 1 :=
  Subsingleton.elim _ _


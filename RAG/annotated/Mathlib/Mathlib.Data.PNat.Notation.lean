/-- `ℕ+` is the type of positive natural numbers. It is defined as a subtype,
  and the VM representation of `ℕ+` is the same as `ℕ` because the proof
  is not stored. -/
def PNat := { n : ℕ // 0 < n } deriving DecidableEq


@[inherit_doc]
notation "ℕ+" => PNat


/-- The underlying natural number -/
@[coe]
def PNat.val : ℕ+ → ℕ := Subtype.val


instance coePNatNat : Coe ℕ+ ℕ :=
  ⟨PNat.val⟩


instance : Repr ℕ+ :=
  ⟨fun n n' => reprPrec n.1 n'⟩

